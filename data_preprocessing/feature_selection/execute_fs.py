from .genetic_algo import ga_feature_selection
from .random_forest import rf_feature_selection
from .recursive_feature_elimination import rfe_feature_selection
from .rfe_cv import rfecv_feature_selection
from .grey_wolf_optimisation import gwo_feature_selection
from .whale_optimisation_algo import woa_feature_selection
from .harris_hawk_optimisation import hho_feature_selection
from .firefly_algo import fa_feature_selection
from .cuckoo_search import cs_feature_selection
from .bat_algo import ba_feature_selection
from .fs_functions import find_best_k
import numpy as np


fs = [ga_feature_selection, rf_feature_selection, rfe_feature_selection, rfecv_feature_selection, gwo_feature_selection,
      woa_feature_selection, hho_feature_selection, fa_feature_selection, cs_feature_selection, ba_feature_selection]

best_features = [None] * len(fs)
models = [None] * len(fs)

print_fs = True

def execute_fs(df, target):
    best_k = find_best_k(df.drop(columns=[target], axis=1), df[target])

    for i, f in enumerate(fs):
        models[i] = str.upper(f.__name__[:-len("_feature_selection")]) + "-"
        np.random.seed(42)
        if f.__name__ == 'ga_feature_selection':
            best_features[i] = list(f(df, target))
            best_features[i].append(target)
        elif f.__name__ == 'rfecv_feature_selection':
            best_features[i] = list(f(df, target, print_fs))
            best_features[i].append(target)
        elif f.__name__ == 'rfe_feature_selection':
            best_features[i] = list(f(df, target))
            best_features[i].append(target)
        elif f.__name__ == 'rf_feature_selection':
            best_features[i] = list(f(df, target, print_fs))
            best_features[i].append(target)
        else:
            best_features[i] = f(df, target, print_fs, best_k)
            best_features[i].append(target)
    
    return best_features, models