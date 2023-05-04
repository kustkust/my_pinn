import inspect

def print_t(t, full: bool = True, debug=False):
    if not debug:
        return
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name,
             var_val in callers_local_vars if var_val is t]
    if len(names) != 1:
        raise
    if full:
        print(f"{names[0]}{t.shape}", t.numpy(), sep='\n')
    else:
        print(f"{names[0]}{t.shape}")