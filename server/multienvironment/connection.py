import socket

class Connection:
    def __init__(self, ip: str | None = None, port: str | None = None) -> None:
        self._ip = ip
        self._port = port
