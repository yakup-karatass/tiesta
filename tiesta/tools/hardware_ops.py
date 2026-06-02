"""
tiesta/tools/hardware_ops.py
────────────────────────────
Hardware operations allowing Tiesta to read serial ports and monitor
embedded systems like ESP32s, Arduinos, or robotic frameworks natively.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from tiesta.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class HardwareOpsTool(BaseTool):
    """Provides embedded systems and hardware debugging capabilities."""

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="list_serial_ports",
                description=(
                    "Scan the physical machine for connected serial/COM devices (e.g., ESP32, "
                    "Arduino, robotics). Returns a list of ports and hardware descriptions."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._handle_list_serial_ports,
            ),
            ToolDefinition(
                name="read_serial_monitor",
                description=(
                    "Open a serial port and capture live streaming logs/output from a connected "
                    "microcontroller for a specified duration."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "port": {
                            "type": "string",
                            "description": "The port to connect to (e.g., COM3 or /dev/ttyUSB0).",
                        },
                        "baudrate": {
                            "type": "integer",
                            "description": "The baud rate (default: 115200).",
                            "default": 115200,
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "How long to listen for logs (default: 5 seconds).",
                            "default": 5,
                        },
                    },
                    "required": ["port"],
                    "additionalProperties": False,
                },
                handler=self._handle_read_serial_monitor,
            ),
        ]

    async def _handle_list_serial_ports(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Scan and list physical serial ports."""
        if serial is None:
            return {"error": "pyserial library is not installed."}

        try:
            ports = serial.tools.list_ports.comports()
            results = []
            for p in ports:
                results.append({
                    "port": p.device,
                    "description": p.description,
                    "hwid": p.hwid,
                })
            return {"ports": results}
        except Exception as exc:
            return {"error": f"Failed to list serial ports: {exc}"}

    async def _handle_read_serial_monitor(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read streaming data from a serial port in a background thread."""
        if serial is None:
            return {"error": "pyserial library is not installed."}

        port = args.get("port", "")
        baudrate = int(args.get("baudrate", 115200))
        timeout_seconds = int(args.get("timeout_seconds", 5))

        if not port:
            return {"error": "Port is required."}

        # Run the blocking serial read in a thread so we don't freeze the TUI
        try:
            loop = asyncio.get_event_loop()
            logs = await loop.run_in_executor(
                None, 
                self._sync_read_serial, 
                port, 
                baudrate, 
                timeout_seconds
            )
            return {"status": "ok", "port": port, "logs": logs}
        except Exception as exc:
            return {"error": f"Failed to read from {port}: {exc}"}

    def _sync_read_serial(self, port: str, baudrate: int, duration: int) -> str:
        """Blocking function to read from the serial port."""
        accumulated_data = bytearray()
        
        # We set a short read timeout so the while loop doesn't block infinitely
        with serial.Serial(port, baudrate, timeout=1) as ser:
            start_time = time.time()
            while time.time() - start_time < duration:
                if ser.in_waiting > 0:
                    accumulated_data.extend(ser.read(ser.in_waiting))
                else:
                    time.sleep(0.1)  # Prevent CPU spinning
                    
        return accumulated_data.decode(errors="replace")
