import numpy as np

# Custom Error Types
class HMMError(Exception):
    pass


class HMMValidationError(HMMError):
    pass

def logsum10(a, axis=None):
    """Compute the log of the sum of power 10 of input elements.

        Args:
            a (array_like): Input array
            axis: Axis or axes over which the sum is taken.
    """
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.power(10 ,a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=False)
        out = np.log10(s)

    a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out