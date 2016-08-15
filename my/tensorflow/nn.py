from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.util import nest

from my.tensorflow import flatten, reconstruct


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    out = reconstruct(flat_out, args[0], 1)
    return out

