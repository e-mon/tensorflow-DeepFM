import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import config
import os

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
def gini_norm(pred, actual):
    if type(actual) is lgb.Dataset:
        actual = actual.get_label()
    return 'gini', gini(actual, pred) / gini(actual, actual), True

def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")

NUMERIC_COLS = [
    # # binary
    # "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
    # "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
    # "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
    # "ps_ind_17_bin", "ps_ind_18_bin",
    # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
    # numeric
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",

    # feature engineering
    "missing_feat", "ps_car_13_x_ps_reg_03",
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]
train_df = pd.read_csv('./data/train.csv', index_col=0)
test_df = pd.read_csv('./data/test.csv', index_col=0)

predictors = list(set(train_df.columns) - set(IGNORE_COLS))
categorical_cols = list(set(predictors) - set(NUMERIC_COLS))

folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(train_df[predictors], train_df['target']))
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth' : 5,
    'num_leaves': 16,
    'colsample_bytree': 0.8,
    'learning_rate': 0.01,
    'verbose': 1,
    'random_seed' : 2018,
}

y_test_meta = np.zeros((train_df.shape[0], 1), dtype=float)
gini_result_cv = np.zeros(len(folds), dtype=float)
for i, (train_idx, test_idx) in enumerate(folds):
    X_train, y_train = train_df.iloc[train_idx], train_df.iloc[train_idx]['target']
    X_valid, y_valid = train_df.iloc[test_idx], train_df.iloc[test_idx]['target']
    
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train[predictors], y_train,
                    feature_name=predictors,
                    categorical_feature = categorical_cols)
    lgvalid = lgb.Dataset(X_valid[predictors], y_valid,
                    feature_name=predictors,
                    categorical_feature = categorical_cols)

    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100,
        feval=gini_norm,
    )
    y_valid_pred = lgb_clf.predict(X_valid[predictors])
    gini_result_cv[i] = gini_norm(y_valid_pred, y_valid)[1]
    y_test_meta = lgb_clf.predict(test_df[predictors])

y_test_meta /= float(len(folds))
filename = "%s_Mean%.5f_Std%.5f.csv"%('lgbm', gini_result_cv.mean(), gini_result_cv.std())
_make_submission(test_df['id'].values, y_test_meta, filename)


