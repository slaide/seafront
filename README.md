# seafront
microscope software stack

this software is mainly intended as an open-source interface for the Cephla SQUID microscope, specifically the HCS version.

The latest official interface is written in python with Qt GUI, which suffers from a few drawbacks (like no builtin scaling for different display resolutions, display code integrated into low level code, etc.).

This new interface has a backend in python, with a web interface on top. This allows microscope interaction over a network, and more fundamental separation of low level control from display functionality.