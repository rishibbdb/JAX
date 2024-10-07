import numpy as np
import jaxlib
import jax.numpy as jnp
from typing import List, Tuple, Any
import os
import jax


def get_network_eval_fn(bpath: str = '../../data/network/', n_layer=9, dtype=jnp.float64):
    """
    """
    params = []

    for i in range(0, n_layer):
        layer_weights = np.load(os.path.join(bpath, f'dense_{i}_weights.npy'))
        layer_bias = np.load(os.path.join(bpath, f'dense_{i}_bias.npy'))
        params.append((
                        jnp.array(layer_weights, dtype=dtype),
                        jnp.array(layer_bias, dtype=dtype)
                    ))

    params = tuple(params)

    def eval_network(x):
        """
        """

        x = jnp.tanh(jnp.dot(x, params[0][0]) + params[0][1])

        # residual block 1
        y = jnp.tanh(jnp.dot(x, params[1][0]) + params[1][1])
        y = jnp.tanh(jnp.dot(y, params[2][0]) + params[2][1])
        x = jnp.tanh(jnp.dot(y, params[3][0]) + params[3][1]) + x

        # residual block 2
        y = jnp.tanh(jnp.dot(x, params[4][0]) + params[4][1])
        y = jnp.tanh(jnp.dot(y, params[5][0]) + params[5][1])
        x = jnp.tanh(jnp.dot(y, params[6][0]) + params[6][1]) + x

        # outputs
        y = jnp.tanh(jnp.dot(x, params[7][0]) + params[7][1])
        z = jnp.dot(y, params[8][0]) + params[8][1]

        return z

    return eval_network


def get_network_eval_v_fn(bpath: str = '../../data/network/', n_layer=9, dtype=jnp.float64):
    """
    """
    eval_network = get_network_eval_fn(bpath = bpath, n_layer=n_layer, dtype=dtype)
    eval_network_v = jax.jit(jax.vmap(eval_network, 0, 0))
    return eval_network_v


# consider code below as deprecated.
# custom classes don't work well with jax autograd.

class TriplePandleNet:
    def __init__(self, bpath: str) -> None:
        self.params = self._get_network_params(bpath)

    def eval(self, x: jaxlib.xla_extension.ArrayImpl) \
            -> jaxlib.xla_extension.ArrayImpl:
        """
        """
        return _eval_network(x, self.params)

    def eval_on_batch(self, x: jaxlib.xla_extension.ArrayImpl) \
            -> jaxlib.xla_extension.ArrayImpl:
        """
        """
        return _eval_network_on_batch(x, self.params)

    def get_network_params(self) -> Tuple[Tuple[jaxlib.xla_extension.ArrayImpl]]:
        return self.params

    def _get_network_params(self, bpath: str) -> Tuple[Tuple[jaxlib.xla_extension.ArrayImpl]]:
        """
        """
        params = []

        for i in range(0, 9):
            layer_weights = np.load(os.path.join(bpath, f'dense_{i}_weights.npy'))
            layer_bias = np.load(os.path.join(bpath, f'dense_{i}_bias.npy'))
            params.append((jnp.array(layer_weights), jnp.array(layer_bias)))

        for param in params:
            param[0].devices()
            param[1].devices()

        return tuple(params)


def _eval_network(x: jaxlib.xla_extension.ArrayImpl,
        params: Tuple[Tuple[jaxlib.xla_extension.ArrayImpl]]) \
                -> jaxlib.xla_extension.ArrayImpl:
    """
    """

    x = jnp.tanh(jnp.dot(x, params[0][0]) + params[0][1])

    # residual block 1
    y = jnp.tanh(jnp.dot(x, params[1][0]) + params[1][1])
    y = jnp.tanh(jnp.dot(y, params[2][0]) + params[2][1])
    x = jnp.tanh(jnp.dot(y, params[3][0]) + params[3][1]) + x

    # residual block 2
    y = jnp.tanh(jnp.dot(x, params[4][0]) + params[4][1])
    y = jnp.tanh(jnp.dot(y, params[5][0]) + params[5][1])
    x = jnp.tanh(jnp.dot(y, params[6][0]) + params[6][1]) + x

    # outputs
    y = jnp.tanh(jnp.dot(x, params[7][0]) + params[7][1])
    z = jnp.dot(y, params[8][0]) + params[8][1]
    return z

_eval_network_on_batch = jax.jit(jax.vmap(_eval_network, (0, None), 0))
