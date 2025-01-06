from typing import Callable

def make_integer(a: int) -> Callable:
    def integer() -> int:
        return a

    return integer

def make_floating_point(a: float) -> Callable:
    def floating_point() -> float:
        return a

    return floating_point

def make_string(a: str) -> Callable:
    def string() -> str:
        return a
    return string