"""Utility functions."""

from typing import Tuple


def dec2hex(decimal: int) -> Tuple[str, str]:
    """Encode decimal to hexadecimal string.

    :Example:

    >>> dec2hex(10)
    ('0a', '0a')
    >>> dec2hex(100000)
    ('18', '186a0')
    """
    hexadecimal = "{:0>2x}".format(decimal)
    return hexadecimal[:2], hexadecimal
