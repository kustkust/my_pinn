import inspect
import tensorflow as tf
import numpy as np

def print_t(t, full: bool = True, debug=False):
    if not debug:
        return
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name,
             var_val in callers_local_vars if var_val is t]
    if len(names) == 0:
        names.append('')
    if full:
        if type(t) is tf.Tensor:
            print(f"{names[0]}{t.shape}", t.numpy(), sep='\n')
        elif type(t) is np.array:
            print(f"{names[0]}{t.shape}", t, sep='\n')
    else:
        print(f"{names[0]}{t.shape}")
