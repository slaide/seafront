#!/usr/bin/env python3
"""
Entry point: pick a microscope from the config, build the Core that drives it, wrap it in the
HTTP app, and serve.

Everything the server can do lives in seafront/server/; this module only decides which
microscope to open and how to listen.
"""

import argparse
import asyncio
import os
import pathlib as path
import signal
import socket
import subprocess
import sys
import threading

import json5
import uvicorn
from fastapi import HTTPException

from seafront.config.basics import GlobalConfigHandler, MicroscopeConfig, ServerConfig
from seafront.config.core_config import UnconfiguredDeviceIdError
from seafront.config.handles import CameraConfig, LaserAutofocusConfig, SystemConfig
from seafront.logger import logger
from seafront.server.app import make_app
from seafront.server.commands import EstablishHardwareConnection
from seafront.server.core import Core


@logger.catch
def main():
    # Load server config to get available microscopes and port for help message
    with GlobalConfigHandler.home_config().open("r") as f:
        server_config = ServerConfig(**json5.load(f))

    available_microscopes = [m.get(SystemConfig.MICROSCOPE_NAME.value, "<unnamed>") for m in server_config.microscopes]
    microscope_list = ", ".join(f'"{name}"' for name in available_microscopes)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seafront Microscope Server")
    parser.add_argument(
        "--microscope",
        type=str,
        help=f"Name of microscope configuration to use. Available: {microscope_list} (defaults to first)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"Port to run the server on (default: {server_config.port} from config)"
    )
    parser.add_argument(
        "--host",
        type=str,
        help=f'Bind address (default: "{server_config.host}" from config). Use "0.0.0.0" or "::" to expose on the network.'
    )
    args = parser.parse_args()

    # Override port/host from config if specified on command line
    if args.port:
        server_config.port = args.port
    if args.host:
        server_config.host = args.host

    if len(server_config.microscopes) == 0:
        logger.critical("No microscope configurations found in config file")
        return

    # Determine which microscope to use
    selected_microscope: MicroscopeConfig | None = None
    if args.microscope:
        # Find microscope by name (using handle key)
        for m in server_config.microscopes:
            if m.get(SystemConfig.MICROSCOPE_NAME.value) == args.microscope:
                selected_microscope = m
                break
        if selected_microscope is None:
            available_names = [m.get(SystemConfig.MICROSCOPE_NAME.value, "<unnamed>") for m in server_config.microscopes]
            logger.critical(f"Microscope '{args.microscope}' not found. Available: {available_names}")
            return
    else:
        # Default to first microscope
        selected_microscope = server_config.microscopes[0]

    # Log startup information
    microscope_name = selected_microscope.get(SystemConfig.MICROSCOPE_NAME.value, "unknown")
    main_camera_id = selected_microscope.get(CameraConfig.MAIN_ID.value, "unknown")
    main_camera_driver = selected_microscope.get(CameraConfig.MAIN_DRIVER.value, "unknown")
    logger.info(f"🔬 Selected microscope: {microscope_name}")
    logger.info(f"📷 Main camera: {main_camera_id} (driver: {main_camera_driver})")
    if selected_microscope.get(LaserAutofocusConfig.AVAILABLE.value) == "yes":
        laf_camera_id = selected_microscope.get(LaserAutofocusConfig.CAMERA_ID.value, "unknown")
        laf_camera_driver = selected_microscope.get(LaserAutofocusConfig.CAMERA_DRIVER.value, "unknown")
        logger.info(f"🎯 Autofocus camera: {laf_camera_id} (driver: {laf_camera_driver})")
    logger.info(f"🌐 Server port: {server_config.port}")

    logger.info(f"initializing core server with microscope: {microscope_name}")

    # Initialize global config with selected microscope
    GlobalConfigHandler.reset(microscope_name)

    # Check for default protocol file, generate if missing
    default_protocol_file = GlobalConfigHandler.home_acquisition_config_dir() / "default.json"

    if not default_protocol_file.exists():
        logger.info(f"Default protocol file not found: {default_protocol_file}")
        logger.info(f"Generating default protocol for microscope '{microscope_name}'...")
        try:
            script_path = (path.Path(__file__).parent.parent / "scripts/generate_default_protocol.py").resolve()
            result = subprocess.run(
                [sys.executable, str(script_path), "--microscope", microscope_name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.critical(f"Failed to generate default protocol: {result.stderr}")
                return
            logger.info(result.stdout.strip())
        except Exception as e:
            logger.critical(f"Failed to generate default protocol: {e}")
            return

        # Re-check that file was created
        if not default_protocol_file.exists():
            logger.critical(f"Default protocol file still not found after generation: {default_protocol_file}")
            return

    logger.info(f"✓ Default protocol found: {default_protocol_file}")

    try:
        core = Core(selected_microscope)
    except UnconfiguredDeviceIdError as e:
        logger.critical(f"startup - {e}")
        return

    app = make_app(core)

    def establish_hardware_connection():
        """
        Connect to the hardware once, at startup, in this process.

        Not over HTTP to ourselves. This is the *cold* connect - the one that opens the
        devices, turns the illumination off and homes the stage - and it belongs to the
        server, not to whichever client happens to ask first. Doing it here is also what
        makes "another process already owns this device" a startup failure again: over HTTP
        it arrived as a status code among others, here it is the exception it always was.

        Runs on its own thread so the server starts listening straight away; homing takes
        a while. The worker serialises it against anything a client sends meanwhile.
        """
        try:
            logger.info("establishing hardware connection at startup")
            asyncio.run(core.run(EstablishHardwareConnection()))
            logger.info("hardware connection established")
        except HTTPException as e:
            if e.status_code == 503:
                # Device is already in use by another process - must exit
                logger.critical(f"hardware connection failed: {e.detail}")
                logger.critical("A hardware device is already in use by another process.")
                logger.critical("Shutting down server...")
                # Send SIGINT to gracefully shut down uvicorn
                os.kill(os.getpid(), signal.SIGINT)
            else:
                logger.warning(f"hardware connection failed: {e.detail}")
                logger.warning("Hardware connection could not be established. You may retry via the UI.")
            return
        except Exception as e:
            logger.warning(f"failed to establish hardware connection at startup: {e}")
            return

        # Warm the hardware limits cache while we are here: it is a one-off camera query, and
        # doing it now keeps it off the first client request that asks what the camera can do.
        try:
            logger.info("initializing hardware limits cache")
            core.get_hardware_capabilities()
            logger.info("hardware limits cache initialized")
        except Exception as e:
            logger.warning(f"failed to initialize hardware limits cache: {e}")

    logger.info("starting http server")

    # Test if port is available before starting uvicorn
    try:
        family = socket.AF_INET6 if ":" in server_config.host else socket.AF_INET
        test_socket = socket.socket(family, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_socket.bind((server_config.host, server_config.port))
        test_socket.close()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            logger.critical(f"Failed to bind to port {server_config.port}. Port is already in use by another process.")
            logger.critical("Please ensure no other instance of Seafront is running, or change the port in your configuration.")
            logger.critical("Exiting...")
            return
        else:
            logger.critical(f"Failed to test port {server_config.port}: {e}")
            logger.critical("Exiting...")
            return

    try:
        # Start hardware connection thread only after we know the port is available
        connection_thread = threading.Thread(target=establish_hardware_connection, daemon=True)
        connection_thread.start()

        # Start uvicorn server
        if server_config.host == "::":
            # uvicorn binds "::" as IPv6-only. Bind a dual-stack socket ourselves
            # (IPV6_V6ONLY=0) so the server is reachable over both IPv6 and IPv4
            # (e.g. both the fe80:: and 169.254.x.x link-local addresses a client
            # may resolve a .local name to).
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            sock.bind((server_config.host, server_config.port))
            uvicorn.Server(
                uvicorn.Config(app, ws_per_message_deflate=False)
            ).run(sockets=[sock])
        else:
            uvicorn.run(
                app,
                host=server_config.host,
                port=server_config.port,
                ws_per_message_deflate=False
            )
        logger.info("http server initialised")

    except Exception as e:
        # This should now rarely be reached due to pre-flight port check
        logger.critical(f"Unexpected server error: {e}")
        logger.critical("Exiting...")
        return
        
    finally:

        logger.info("shutting down")

        core.close()

        logger.info("shutdown complete")


if __name__ == "__main__":
    main()
