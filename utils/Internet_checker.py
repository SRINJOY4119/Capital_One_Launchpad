import socket
from typing import Optional


class InternetChecker:
    """
    A modular class to check internet connectivity.
    """

    def __init__(self, host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0):
        """
        Initialize the checker.
        
        :param host: Remote host to connect to (default: Google DNS).
        :param port: Port number to connect to (default: 53 for DNS).
        :param timeout: Timeout in seconds.
        """
        self.host = host
        self.port = port
        self.timeout = timeout

    def is_connected(self) -> bool:
        """
        Check if the internet is connected.
        
        :return: True if connected, False otherwise.
        """
        try:
            socket.setdefaulttimeout(self.timeout)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
            return True
        except (socket.timeout, socket.error):
            return False

    def get_status_message(self) -> str:
        """
        Get a human-readable status message.
        
        :return: Status string.
        """
        return "Internet is connected " if self.is_connected() else "No internet connection "

    def test_custom_server(self, host: str, port: int, timeout: Optional[float] = None) -> bool:
        """
        Test connectivity to a custom server.
        
        :param host: Host to test.
        :param port: Port to test.
        :param timeout: Optional timeout override.
        :return: True if reachable, False otherwise.
        """
        try:
            socket.setdefaulttimeout(timeout or self.timeout)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
            return True
        except (socket.timeout, socket.error):
            return False


if __name__ == "__main__":
    checker = InternetChecker()
    print(checker.get_status_message())

    if checker.test_custom_server("1.1.1.1", 53):
        print("Cloudflare DNS is reachable ")
    else:
        print("Cloudflare DNS is not reachable ")
