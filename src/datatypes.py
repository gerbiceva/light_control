from typing import Callable
from jax.typing import ArrayLike

Vector2D = ArrayLike
Color = ArrayLike
Vector3D = ArrayLike
ColorArray = ArrayLike
Array = ArrayLike
Curve = Callable

def node(f):
    f.__is_node__ = True
    return f