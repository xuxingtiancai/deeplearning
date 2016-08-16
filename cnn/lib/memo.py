from functools import wraps

def memo(fn):
    @wraps(fn)
    def wrapper(*args):
        result = wrapper.cache.get(args)
        if result is None:
            result = fn(*args)
            wrapper.cache[args] = result
        return result
    wrapper.cache = {}
    return wrapper

def list_memo(fn):
    @wraps(fn)
    def wrapper(list_arg, *args):
        miss_arg = []
        for arg in list_arg:
            if not arg in wrapper.cache:
                miss_arg.append(arg)
        
        fn_result = fn(miss_arg, *args)
        miss_dic = dict(zip(miss_arg, fn_result))
        wrapper.cache.update(miss_dic)
        return [wrapper.cache[arg] for arg in list_arg]
    wrapper.cache = {}
    return wrapper
