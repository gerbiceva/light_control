from jax.typing import ArrayLike
from sympy import Expr

Int = int
Float = float
String = str
Vector2D = ArrayLike
Color = ArrayLike
Vector3D = ArrayLike
ColorArray = ArrayLike
Array = ArrayLike
Curve = Expr
Array2D = ArrayLike
Curve2D = Expr

def node(f):
    f.__is_node__ = True
    return f

def initialize(f):
    f.__initialize__ = True
    return f

def primitive(f):
    f.__primitive__ = True
    return f

def each_tick(f):
    f.__each_tick__ = True
    return f

def thread(f):
    f.__thread__ = True
    return f

def generator(f):
    f.__generator__ = True
    return f

def required(num):
    def wrap(f):
        f.__required__ = num
        return f
    return wrap