"""Utility functions."""

from typing import Tuple


def dec2hex(decimal: int) -> Tuple[str, str]:
    """Encode decimal to hexadecimal string.

    :Example:

    >>> dec2hex(10)
    ('00', '00', '00a')
    >>> dec2hex(100000)
    ('00', '18', '6a0')
    """
    hexadecimal = "{:0>7x}".format(decimal)
    return hexadecimal[:2], hexadecimal[2:4], hexadecimal[4:7]
