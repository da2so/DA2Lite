
import DA2Lite.compression.pruning as pruning

def load_strategy(strategy_name, group_set, pruning_ratio):

    try: 
        strategy_class = getattr(pruning, strategy_name)
    except:
        raise ValueError(f'Invalid strategy name: {strategy_name}')


    strategy = strategy_class(group_set, pruning_ratio)
    
    return strategy



def load_criteria(criteria_name):

    try: 
        criteria_class = getattr(pruning, criteria_name)
    except:
        raise ValueError(f'Invalid strategy name: {criteria_name}')


    criteria = criteria_class()
    
    return criteria
