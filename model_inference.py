import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import json

path_to_inference_data = './rawdata/inference.csv'
model_weight_path = './model_weight.txt'

# 注意：提供的CSV文件内需要包含下列字段与对应的值（需要严格按顺序）
headers = ['STATE_CODE', 'SHRP_ID', 'survey_date', 'REAL_SS_MATL', 
          'GS_THICKNESS', 'REAL2_GS_MATL', 'TS_THICKNESS', 'REAL_TS_MATL', 
          'GB_THICKNESS', 'REAL_GB_MATL', 'TB_THICKNESS', 'REAL_TB_MATL', 
          'PC_THICKNESS', 'AC_THICKNESS', 'REAL_AC_MATL', 'REAL_Construction_Type', 
          'LANE_WIDTH', 'LANE_LENGTH', 'LANES_NO', 'PRECIPITATION', 'EVAPORATION', 
          'EMISSIVITY_AVG', 'SHORTWAVE_SURFACE_AVG', 'REL_HUM_AVG_AVG', 'TEMP_AVG', 
          'DAYS_ABOVE_32_C', 'DAYS_BELOW_0_C', 'FREEZE_INDEX', 'FREEZE_THAW', 
          'AADT_ALL_VEHIC_2WAY', 'AADT_TRUCK_COMBO_2WAY', 'AADT_ALL_VEHIC', 
          'AADT_TRUCK_COMBO', 'ANL_KESAL_LTPP_LN_YR', 'prev_PCI']

features = ['survey_date', 'REAL_SS_MATL', 'GS_THICKNESS', 'REAL2_GS_MATL', 'TS_THICKNESS', 'REAL_TS_MATL', 
           'GB_THICKNESS', 'REAL_GB_MATL', 'TB_THICKNESS', 'REAL_TB_MATL', 'PC_THICKNESS', 'AC_THICKNESS', 
           'REAL_AC_MATL', 'REAL_Construction_Type', 'LANE_WIDTH', 'LANE_LENGTH', 'LANES_NO', 'PRECIPITATION', 
           'EVAPORATION', 'EMISSIVITY_AVG', 'SHORTWAVE_SURFACE_AVG', 'REL_HUM_AVG_AVG', 'TEMP_AVG', 
           'DAYS_ABOVE_32_C', 'DAYS_BELOW_0_C', 'FREEZE_INDEX', 'FREEZE_THAW', 'AADT_ALL_VEHIC_2WAY', 
           'AADT_TRUCK_COMBO_2WAY', 'AADT_ALL_VEHIC', 'AADT_TRUCK_COMBO', 'ANL_KESAL_LTPP_LN_YR', 'prev_PCI']

class_map = {
    'REAL_SS_MATL': {'Sandstone': 0,
                'Rock': 1,
                'Coarse-Grained Soil': 2,
                'Other': 3,
                'Fine-Grained Soil': 4},
    'REAL2_GS_MATL': {'Sand': 0,
                    'Coarse-Grained Soil': 1,
                    'Other': 2,
                    'Fine-Grained Soil': 3,
                    'Soil-Aggregate Mixture': 4,
                    'No Layer': 5,
                    'Crushed Stone': 6},
    'REAL_TS_MATL': {'Cement Treated': 0,
                    'Pozzolanic Aggregate Treated': 1,
                    'Other': 2,
                    'Asphalt Treated': 3,
                    'Cement Treated+Asphalt Treated': 4,
                    'No Layer': 5,
                    'Lime Treated': 6},
    'REAL_GB_MATL': {'Coarse-Grained Soil': 0,
                    'Other': 1,
                    'Fine-Grained Soil': 2,
                    'Soil-Aggregate Mixture (Predominantly Coarse-Grained)': 3,
                    'No Layer': 4,
                    'Soil-Aggregate Mixture (Predominantly Fine-Grained)': 5,
                    'Crushed Stone': 6},
    'REAL_TB_MATL': {'Cement Treated': 0,
                    'Pozzolanic Aggregate Treated': 1,
                    'Other': 2,
                    'Asphalt Treated': 3,
                    'Cement Treated+Asphalt Treated': 4,
                    'No Layer': 5,
                    'Lime Treated': 6},
    'REAL_AC_MATL': {'Sand Asphalt': 0,
                    'Recycled Cold Mixed AC': 1,
                    'Hot Mixed AC': 2,
                    'Recycled Hot Mixed AC': 3,
                    'Other': 4,
                    'Hot Mixed OGFC': 5,
                    'No Layer': 6,
                    'Cold Mixed AC': 7,
                    'Warm Mixed AC': 8},
    'REAL_Construction_Type': {'No_construction': 0,
                            'Crack Sealing': 1,
                            'Mill and Overlay': 2,
                            'Overlay': 3,
                            'Surface Treatment': 4,
                            'Patch': 5,
                            'Seal Coat': 6}
}

target = 'PCI'

def get_interval_number(value, interval_length=10):

    if value < 0 or value > 100:
        raise ValueError("Value must be between 0 and 100")
    
    interval_number = int(value // interval_length)
    
    if interval_number >= 10:
        interval_number = 9
    
    return interval_number

def main():
    all_data = dict()
    with open(path_to_inference_data, "r", encoding="utf-8") as file:

        csv_reader = csv.reader(file)

        header = next(csv_reader)

        for row in csv_reader:
            road_id = row[0] + '_' + row[1]
            row[1] = road_id
            if road_id not in all_data.keys():
                all_data[road_id] = []
            all_data[road_id].append(row[1:])
    file.close()

    for key in all_data.keys():
        sub_data = sorted(all_data[key], key=lambda x:int(x[1]))
        start = int(sub_data[0][1])
        sub_data = [[row[0], int(row[1])-start] + row[2:] for row in sub_data]
        all_data[key] = sub_data

    inputs = {key:[] for key in headers[1:]}

    for road_id, road_name in enumerate(list(all_data.keys())):
        for row in all_data[road_name]:
            inputs['SHRP_ID'].append(road_id)
            for key, value in zip(headers[2:], row[1:]):
                if key in class_map.keys():
                    inputs[key].append(class_map[key][value])
                elif key == headers[2]:
                    inputs[key].append(int(value))
                else:
                    inputs[key].append(float(value) if value != '' else None)

    inputs['prev_PCI'] = [get_interval_number(x) for x in inputs['prev_PCI'].tolist()]
    inference_df = pd.DataFrame(inputs)

    inference_model = lgb.Booster(model_file=model_weight_path)

    all_predicted_PCI = []
    idx = 0
    while idx < len(inference_df):
        data_loc = inference_df.loc[idx]
        predicted_PCI = inference_model.predict(data_loc[features])
        all_predicted_PCI.append(predicted_PCI)
    
    return all_predicted_PCI

if __name__ == '__main__':
    results = main()
