import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import json

path_to_train_data = './rawdata/filled_interval_all.json'
weight_output = './model_weight.txt'

features = ['survey_date', 'REAL_SS_MATL', 'GS_THICKNESS', 'REAL2_GS_MATL', 'TS_THICKNESS', 'REAL_TS_MATL', 
           'GB_THICKNESS', 'REAL_GB_MATL', 'TB_THICKNESS', 'REAL_TB_MATL', 'PC_THICKNESS', 'AC_THICKNESS', 
           'REAL_AC_MATL', 'REAL_Construction_Type', 'LANE_WIDTH', 'LANE_LENGTH', 'LANES_NO', 'PRECIPITATION', 
           'EVAPORATION', 'EMISSIVITY_AVG', 'SHORTWAVE_SURFACE_AVG', 'REL_HUM_AVG_AVG', 'TEMP_AVG', 
           'DAYS_ABOVE_32_C', 'DAYS_BELOW_0_C', 'FREEZE_INDEX', 'FREEZE_THAW', 'AADT_ALL_VEHIC_2WAY', 
           'AADT_TRUCK_COMBO_2WAY', 'AADT_ALL_VEHIC', 'AADT_TRUCK_COMBO', 'ANL_KESAL_LTPP_LN_YR']
target = 'PCI'
group_col = 'SHRP_ID'

def preprocess(df):
    df = df.sort_values([group_col, 'survey_date'])
    df['prev_PCI'] = df.groupby(group_col)[target].shift(1)
    df = df[~df['prev_PCI'].isna()]
    df = df[~df[target].isna()]
    return df

def main():

    with open(path_to_train_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    processed_df = preprocess(df.copy())

    features.append('prev_PCI')

    group_kfold = GroupKFold(n_splits=5)
    groups = processed_df[group_col]

    X = processed_df[features]
    y = processed_df[target]

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1
    }

    val_scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50)]
        )
        
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_scores.append(val_rmse)
        print(f'Validation RMSE: {val_rmse:.4f}')

    print(f'Mean Validation RMSE: {np.mean(val_scores):.4f}')

    model.save_model(weight_output)

    print('测试例子：')
    val = processed_df.loc[3]
    predicted_PCI = model.predict(val[features])
    print('PCI Ground Truth:', int(val['PCI']), '   Predicted PCI:', round(predicted_PCI[0]))

if __name__ == '__main__':
    main()
