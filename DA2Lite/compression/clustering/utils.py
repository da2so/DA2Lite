
from DA2Lite.compression.clustering import methods


def load_method(method_name, **kwargs):

    try: 
        method_class = getattr(methods, method_name)
    except:
        raise ValueError(f'Invalid method name: {method_name}')

    method = method_class(**kwargs)
    
    return method

def load_strategy(strategy_name, group_set, pruning_ratio):

    try: 
        strategy_class = getattr(methods, strategy_name)
    except:
        raise ValueError(f'Invalid strategy name: {strategy_name}')

    strategy = strategy_class(group_set, pruning_ratio)
    
    return strategy

