import timeit

import jax.numpy as jnp
from jax import jit, random


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


if __name__ == "__main__":
    setup = """
from __main__ import selu
import jax.numpy as jnp
from jax import jit, random
key = random.PRNGKey(0)
x = random.normal(key, (1000000,))
"""

    setup_jit = """
from __main__ import selu
import jax.numpy as jnp
from jax import jit, random
key = random.PRNGKey(0)
x = random.normal(key, (1000000,))
selu_jit = jit(selu)
"""

    number_trials = 10000
    timeit_result = timeit.timeit("selu(x)", setup=setup, number=number_trials)
    print(f"normal: {timeit_result / number_trials}")

    timeit_result = timeit.timeit("selu_jit(x)", setup=setup_jit, number=number_trials)
    print(f"JIT: {timeit_result / number_trials}")
