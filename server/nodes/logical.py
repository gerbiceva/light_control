from datatypes import Float, Array, node
import jax.numpy as jnp

@node
def where(test: Array, this: Array, that: Array) -> Array:
    """
    Where

    Where test is positive this, where it's negative that.

    Parameters
    ----------
    test : Array
        Test array.

    this : Array
        This array.

    That : Array
        That array.

    Returns
    -------
    array : Array
        Output array.
    """
    return jnp.where(test >= 0, this, that)