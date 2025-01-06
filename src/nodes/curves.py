from datatypes import Curve, Array
import jax
import jax.numpy as jnp
from jax import lax


def sample_curve(curve: Curve, x: float) -> float:
    return curve(x)


def sample_array(curve: Curve, resolution: int) -> Array:
    return jax.vmap(curve, in_axes=0, out_axes=0)(jnp.linspace(0, 1, resolution))


def pad_curve(curve: Curve, padding: float) -> Curve:
    def new_curve(x: float) -> float:
        return lax.cond(
            x < (1 - padding),
            lambda _: curve(x / (1 - padding)),
            lambda _: 0.0,
            operand=None,
        )

    return new_curve


def move_curve(curve: Curve, move: float) -> Curve:
    def new_curve(x: float) -> float:
        return curve((x + move) % 1)

    return new_curve


def linear_curve() -> Curve:
    def new_curve(x: int) -> int:
        return x

    return new_curve
