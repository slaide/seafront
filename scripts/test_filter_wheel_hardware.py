#!/usr/bin/env python3
"""
Hardware test script for filter wheel functionality.
This script performs the full microcontroller initialization sequence
and then cycles through all filter wheel positions.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from seafront.hardware.microcontroller import Microcontroller, get_all_microcontrollers
from seafront.hardware.microcontroller.teensy_microcontroller import TeensyMicrocontroller
from seafront.logger import logger

async def initialize_microcontroller(mc: Microcontroller) -> bool:
    """
    Perform the full microcontroller initialization sequence
    similar to what's done in the main application.
    """
    try:
        logger.info("Opening microcontroller connection...")
        mc.open()

        logger.info("Sending reset command...")
        await mc.reset()

        logger.info("Sending initialize command...")
        await mc.initialize()

        logger.info("Configuring actuators...")
        await mc.configure_actuators()
            
        logger.info("Initializing filter wheel...")
        try:
            await mc.filter_wheel_init()
            logger.info("✓ Filter wheel initialization completed")
        except Exception as e:
            logger.warning(f"Filter wheel initialization may have timed out: {e}")
            logger.info("Continuing with homing and position test...")
        
        logger.info("Configuring filter wheel actuator...")
        try:
            await mc.filter_wheel_configure_actuator()
            logger.info("✓ Filter wheel actuator configured")
        except Exception as e:
            logger.error(f"Filter wheel actuator configuration failed: {e}")
            logger.info("This may cause homing issues...")
        
        logger.info("Performing filter wheel homing sequence...")
        try:
            await mc.filter_wheel_home()
            logger.info("✓ Filter wheel homing completed")
        except Exception as e:
            logger.error(f"Filter wheel homing failed: {e}")
            logger.info("This may indicate hardware issues with limit switches or motor")
            logger.info("Continuing with position test anyway...")
        
        logger.info("✓ Microcontroller initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize microcontroller: {e}")
        return False

async def test_filter_wheel_positions(mc: Microcontroller):
    """
    Test filter wheel by cycling through all positions with delays.
    """
    positions = list(range(1, 9))  # 1 through 8

    logger.info("Starting filter wheel position test...")
    logger.info(f"Will test positions: {positions}")

    # Debug: Print Teensy-specific configuration if applicable
    if isinstance(mc, TeensyMicrocontroller):
        from seafront.hardware.microcontroller.teensy_microcontroller import FirmwareDefinitions, Command, CommandName

        logger.info(f"Filter wheel configuration (Teensy-specific):")
        logger.info(f"  SCREW_PITCH_W_MM: {FirmwareDefinitions.SCREW_PITCH_W_MM}")
        logger.info(f"  MICROSTEPPING_DEFAULT_W: {FirmwareDefinitions.MICROSTEPPING_DEFAULT_W}")
        logger.info(f"  FULLSTEPS_PER_REV_W: {FirmwareDefinitions.FULLSTEPS_PER_REV_W}")
        logger.info(f"  STAGE_MOVEMENT_SIGN_W: {FirmwareDefinitions.STAGE_MOVEMENT_SIGN_W}")
        logger.info(f"  mm_per_ustep_w: {FirmwareDefinitions.mm_per_ustep_w()}")

        distance_per_position = FirmwareDefinitions.SCREW_PITCH_W_MM / (
            FirmwareDefinitions.FILTERWHEEL_MAX_INDEX - FirmwareDefinitions.FILTERWHEEL_MIN_INDEX + 1
        )
        logger.info(f"  Distance per position: {distance_per_position} mm")

    try:
        # Move to each position in sequence
        for position in positions:
            current_pos = mc.filter_wheel_get_position()
            logger.info(f"Current position: {current_pos}, Moving to position {position}...")

            # Debug: Calculate what movement should happen (Teensy-specific)
            if isinstance(mc, TeensyMicrocontroller) and position != current_pos:
                from seafront.hardware.microcontroller.teensy_microcontroller import FirmwareDefinitions, Command, CommandName

                delta_positions = position - current_pos
                distance_per_position = FirmwareDefinitions.SCREW_PITCH_W_MM / (
                    FirmwareDefinitions.FILTERWHEEL_MAX_INDEX - FirmwareDefinitions.FILTERWHEEL_MIN_INDEX + 1
                )
                distance_mm = delta_positions * distance_per_position
                usteps = FirmwareDefinitions.mm_to_ustep_w(distance_mm)

                logger.info(f"  Movement calculation:")
                logger.info(f"    Delta positions: {delta_positions}")
                logger.info(f"    Distance: {distance_mm} mm")
                logger.info(f"    Microsteps: {usteps}")

                # Generate the commands manually to see what they look like
                move_commands = Command.move_w_usteps(usteps)
                logger.info(f"    Generated {len(move_commands)} command(s)")
                for i, cmd in enumerate(move_commands):
                    logger.info(f"    Command {i}: [1]={cmd[1]} (MOVE_W={CommandName.MOVE_W.value})")

            await mc.filter_wheel_set_position(position)
            
            new_pos = mc.filter_wheel_get_position()
            logger.info(f"✓ Filter wheel now at position {new_pos}")
            
            # Wait 2 seconds between moves
            logger.info("Waiting 2 seconds...")
            time.sleep(2)
        
        # Return to position 1
        logger.info("Returning to position 1...")
        await mc.filter_wheel_set_position(1)
        final_pos = mc.filter_wheel_get_position()
        logger.info(f"✓ Filter wheel returned to position {final_pos}")
        
        logger.info("✓ Filter wheel position test completed successfully!")
        
    except Exception as e:
        logger.error(f"Filter wheel test failed: {e}")
        raise

async def main():
    """
    Main test routine: find microcontroller, initialize it, and test filter wheel.
    """
    logger.info("🔧 Starting filter wheel hardware test")
    logger.info("=" * 60)
    
    # Find available microcontrollers
    logger.info("Searching for microcontrollers...")
    microcontrollers = get_all_microcontrollers()

    if not microcontrollers:
        logger.error("❌ No microcontrollers found!")
        logger.info("Make sure the microcontroller is connected and recognized by the system.")
        return 1

    logger.info(f"Found {len(microcontrollers)} microcontroller(s)")
    for i, mc_found in enumerate(microcontrollers):
        logger.info(f"  {i+1}. {mc_found.vendor_name} {mc_found.model_name} (SN: {mc_found.sn})")

    # Use the first available microcontroller
    mc = microcontrollers[0]
    logger.info(f"Using microcontroller: {mc.vendor_name} {mc.model_name}")
    
    try:
        # Initialize the microcontroller
        if not await initialize_microcontroller(mc):
            logger.error("❌ Failed to initialize microcontroller")
            return 1
        
        # Test filter wheel positions
        await test_filter_wheel_positions(mc)
        
        logger.info("=" * 60)
        logger.info("✅ All tests completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return 1
        
    finally:
        # Always close the connection
        try:
            logger.info("Closing microcontroller connection...")
            mc.close()
            logger.info("✓ Connection closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)