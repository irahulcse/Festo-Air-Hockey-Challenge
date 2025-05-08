import socket
import struct
from contextlib import AbstractContextManager
from typing import Optional


class UDPConnector(AbstractContextManager):
    """
    Tiny helper to push 40‑byte set‑point datagrams to a PLC.

    Packet layout  <BB6xdddd
        enable       : uint8   (0/1)
        acknowledge  : uint8   (0/1)
        padding      : 6 bytes
        velocity     : float64
        acceleration : float64
        x            : float64
        y            : float64
    """

    _FMT  = '<BB6xdddd'
    _SIZE = struct.calcsize(_FMT)

    # ----------------------------------------------------------------- setup
    def __init__(self, plc_ip: str = '192.168.4.201', plc_port: int = 3001) -> None:
        self._plc_addr = (plc_ip, plc_port)

        # new socket each run → no stale queue
        self._sock: Optional[socket.socket] = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(('', 0))  # any free source port
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, UDPConnector._SIZE)

        self.setpoints()  # defaults

    def setpoints(self,
                  enable: bool = True,
                  acknowledge: bool = False,
                  velocity: float = 0.0,
                  acceleration: float = 0.0) -> None:
        """Store set‑point fields (sent with every coordinate update)."""
        self._enable       = 1 if enable else 0
        self._acknowledge  = 1 if acknowledge else 0
        self._velocity     = float(velocity)
        self._acceleration = float(acceleration)

    # --------------------------------------------------------------- runtime
    def send_coordinates(self, x: float, y: float) -> None:
        """Fire one datagram at the PLC (stateless, no retry)."""
        payload = struct.pack(UDPConnector._FMT,
                              self._enable,
                              self._acknowledge,
                              self._velocity,
                              self._acceleration,
                              x, y)
        if self._sock:
            self._sock.sendto(payload, self._plc_addr)

    # -------------------------------------------------------------- teardown
    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    # context‑manager hooks
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
