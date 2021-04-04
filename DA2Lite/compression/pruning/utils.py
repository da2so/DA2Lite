
from DA2Lite.compression import pruning

def load_strategy(strategy_name, group_set, pruning_ratio):

    try: 
        strategy_class = getattr(pruning, strategy_name)
    except:
        raise ValueError(f'Invalid strategy name: {strategy_name}')

    strategy = strategy_class(group_set, pruning_ratio)
    
    return strategy

def load_criteria(criteria_name, **kwargs):

    try: 
        criteria_class = getattr(pruning, criteria_name)
    except:
        raise ValueError(f'Invalid criteria name: {criteria_name}')

    criteria = criteria_class(**kwargs)
    
    return criteria


