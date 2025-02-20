import numpy as np
import json
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

features = ['survey_date', 'REAL_SS_MATL', 'GS_THICKNESS', 'REAL2_GS_MATL', 'TS_THICKNESS', 'REAL_TS_MATL', 
           'GB_THICKNESS', 'REAL_GB_MATL', 'TB_THICKNESS', 'REAL_TB_MATL', 'PC_THICKNESS', 'AC_THICKNESS', 
           'REAL_AC_MATL', 'REAL_Construction_Type', 'LANE_WIDTH', 'LANE_LENGTH', 'LANES_NO', 'PRECIPITATION', 
           'EVAPORATION', 'EMISSIVITY_AVG', 'SHORTWAVE_SURFACE_AVG', 'REL_HUM_AVG_AVG', 'TEMP_AVG', 
           'DAYS_ABOVE_32_C', 'DAYS_BELOW_0_C', 'FREEZE_INDEX', 'FREEZE_THAW', 'AADT_ALL_VEHIC_2WAY', 
           'AADT_TRUCK_COMBO_2WAY', 'AADT_ALL_VEHIC', 'AADT_TRUCK_COMBO', 'ANL_KESAL_LTPP_LN_YR', 'prev_PCI']
target = 'PCI'
train_data = './rawdata/filled_interval_all.json'

with open(train_data, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
X = df[features]
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)

params_low = {
    'objective': 'quantile',
    'alpha': 0.45,
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,     
    'num_leaves': 40,          
    'max_depth': 4,            
    'min_data_in_leaf': 20,    
    'lambda_l2': 0.1,          
    'verbose': -1,
    'num_boost_round': 10000    
}
model_low = lgb.train(params_low, train_data)


params_high = {
    'objective': 'quantile',
    'alpha': 0.55,
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,     
    'num_leaves': 40,          
    'max_depth': 4,           
    'min_data_in_leaf': 20,
    'lambda_l2': 0.1,     
    'verbose': -1,
    'num_boost_round': 10000  
}
model_high = lgb.train(params_high, train_data)

pred_low = model_low.predict(X_test)
pred_high = model_high.predict(X_test)

coverage = np.mean((y_test >= pred_low) & (y_test <= pred_high))
print(f"预测区间覆盖率: {coverage * 100:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:50], label="True E")
plt.fill_between(range(50), pred_low[:50], pred_high[:50], alpha=0.3, color="orange", label="Predicted Interval")
plt.legend()
plt.title("LightGBM Predicted Interval")
plt.show()
