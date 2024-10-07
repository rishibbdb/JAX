import numpy as np
import jaxlib
import jax.numpy as jnp
from typing import List, Tuple, Any
import os
import jax

def get_network_eval_fn(bpath: str = '../../data/network/', n_layer=7, dtype=jnp.float64):
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
        x = jnp.tanh(jnp.dot(y, params[2][0]) + params[2][1]) + x

        # residual block 2
        y = jnp.tanh(jnp.dot(x, params[3][0]) + params[3][1])
        x = jnp.tanh(jnp.dot(y, params[4][0]) + params[4][1]) + x

        # outputs
        y = jnp.tanh(jnp.dot(x, params[5][0]) + params[5][1])
        z = jnp.dot(y, params[6][0]) + params[6][1]

        return z

    return eval_network


def get_network_eval_v_fn(bpath: str = '../../data/network/', n_layer=7, dtype=jnp.float64):
    """
    """
    eval_network = get_network_eval_fn(bpath = bpath, n_layer=n_layer, dtype=dtype)
    eval_network_v = jax.jit(jax.vmap(eval_network, 0, 0))
    return eval_network_v
