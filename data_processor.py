import csv
import json
import pandas as pd
import math
from tqdm import tqdm

data_type = 'train' # 如果是推理数据的话就是 test
path_to_rawdata = './rawdata/rawdata.csv'
path_to_quality = './rawdata/quality.csv'
json_output = './rawdata/filled_interval_all.json'

# 录入表头，作为训练数据字典的key
headers = ['STATE_CODE', 'SHRP_ID', 'survey_date', 'PCI', 'REAL_SS_MATL', 
          'GS_THICKNESS', 'REAL2_GS_MATL', 'TS_THICKNESS', 'REAL_TS_MATL', 
          'GB_THICKNESS', 'REAL_GB_MATL', 'TB_THICKNESS', 'REAL_TB_MATL', 
          'PC_THICKNESS', 'AC_THICKNESS', 'REAL_AC_MATL', 'REAL_Construction_Type', 
          'LANE_WIDTH', 'LANE_LENGTH', 'LANES_NO', 'PRECIPITATION', 'EVAPORATION', 
          'EMISSIVITY_AVG', 'SHORTWAVE_SURFACE_AVG', 'REL_HUM_AVG_AVG', 'TEMP_AVG', 
          'DAYS_ABOVE_32_C', 'DAYS_BELOW_0_C', 'FREEZE_INDEX', 'FREEZE_THAW', 
          'AADT_ALL_VEHIC_2WAY', 'AADT_TRUCK_COMBO_2WAY', 'AADT_ALL_VEHIC', 'AADT_TRUCK_COMBO', 'ANL_KESAL_LTPP_LN_YR']

# 主要是将一些文字的类别替换为数字
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
group_col = 'SHRP_ID'

def new_road(df):
    def new_road_group(group):
        rows = group.to_dict('records')
        non_zero_indices = [i for i, row in enumerate(rows) if row['REAL_Construction_Type'] != 0]
        offset = 0
        
        for i in non_zero_indices:
            current = i + offset
            new_row = rows[current].copy()
            rows.insert(current + 1, new_row)
            
            for j in range(current + 1, len(rows)):
                rows[j]['SHRP_ID'] = f"{rows[j]['SHRP_ID']}_fixed"
            
            offset += 1
        
        return pd.DataFrame(rows)
        
    processed_dfs = []
    for shrp_id, group in df.groupby('SHRP_ID'):
        processed_dfs.append(new_road_group(group))

    final_df = pd.concat(processed_dfs).reset_index(drop=True)
    return final_df

def full_fill_data(df):
    tqdm.pandas()

    grouped = df.groupby('SHRP_ID', group_keys=False)
    
    def process_group(group):
        group = group.sort_values('survey_date')
        new_rows = []
        
        for i in range(len(group)):
            current_row = group.iloc[i]
            if i == 0:
                new_row = current_row.copy()
                new_row['Delta_Year'] = 0
                new_row['prev_PCI'] = current_row['PCI']
                new_rows.append(new_row)
            else:
                prev_rows = group.iloc[:i]
                for _, prev_row in prev_rows.iterrows():
                    new_row = current_row.copy()
                    new_row['Delta_Year'] = current_row['survey_date'] - prev_row['survey_date']
                    new_row['prev_PCI'] = prev_row['PCI']
                    new_rows.append(new_row)
        
        return pd.DataFrame(new_rows)
    
    # result = grouped.apply(process_group)
    result = grouped.progress_apply(process_group)
    
    return result

# 把PCI精确值转变为区间值，目前为10一个区间，从0到9
def get_interval_number(value, interval_length=10):

    if value < 0 or value > 100:
        raise ValueError("Value must be between 0 and 100")
    
    interval_number = int(value // interval_length)
    
    if interval_number >= 10:
        interval_number = 9
    
    return interval_number

# 用线性差值的方式处理PCI缺失值
def fill_pci_linear(group):
    group = group.copy().reset_index(drop=True)
    n = len(group)
    segments = []
    i = 0
    
    while i < n:
        row = group.iloc[i]
        if row['Construction'] == 0 and pd.isna(row['PCI']):
            j = i - 1
            prev_pci = None
            while j >= 0:
                if not pd.isna(group.iloc[j]['PCI']):
                    prev_pci = group.iloc[j]['PCI']
                    break
                j -= 1
            if prev_pci is None:
                i += 1
                continue
            
            start_idx = i
            end_idx = i
            while end_idx < n and group.iloc[end_idx]['Construction'] == 0 and pd.isna(group.iloc[end_idx]['PCI']):
                end_idx += 1
            end_idx -= 1 
            
            if end_idx + 1 < n and group.iloc[end_idx + 1]['Construction'] != 0:
                end_pci = group.iloc[end_idx + 1]['PCI']
                if pd.isna(end_pci):
                    end_pci = 10.0
            else:
                end_pci = 10.0
            
            if end_pci > prev_pci:
                end_pci = 10.0
            end_pci = max(end_pci, 10.0)  
            
            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'prev_pci': prev_pci,
                'end_pci': end_pci
            })
            i = end_idx + 1
        else:
            i += 1
    
    for seg in segments:
        start_idx = seg['start_idx']
        end_idx = seg['end_idx']
        prev_pci = seg['prev_pci']
        end_pci = seg['end_pci']
        num_rows = end_idx - start_idx + 1
        
        total_decrease = prev_pci - end_pci
        step = total_decrease / (num_rows + 1)
        
        for i in range(num_rows):
            current_idx = start_idx + i
            current_pci = prev_pci - step * (i + 1)
            current_pci = max(round(current_pci, 2), 10.0)
            group.at[current_idx, 'PCI'] = current_pci
    
    return group

# used_data主要是为了后面可以筛选出高质量数据用
used_data = set()
with open(path_to_quality, "r", encoding="gbk") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        if row[2] == '1':
            used_data.add(row[0] + '_' + row[1])       
file.close()

# High_quality：筛选质量为1的数据
# Filter_none：剔除空值（一整行数据）
# Filled_interval：是否将PCI处理为区间值
High_quality = False
Filter_none = False
Filled_interval = True
all_data = dict()

# 读取原始数据
with open(path_to_rawdata, "r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    print('生成道路ID索引：')
    for row in tqdm(list(csv_reader)):
        road_id = row[0] + '_' + row[1] # 这里把第一列和第二列合并作为路段索引，可以保证唯一性
        if road_id not in used_data and High_quality:
            continue
        if '' in row and Filter_none:
            continue
        row[1] = road_id
        if road_id not in all_data.keys():
            all_data[road_id] = []
        all_data[road_id].append(row[1:])
file.close()

# 生成训练数据，格式为字典，为了方便导出为json并被模型读取
print('生成训练数据：')
inputs = {key:[] for key in headers[1:]}
for road_id, road_name in enumerate(tqdm(list(all_data.keys()))):
    for row in all_data[road_name]:
        # inputs['SHRP_ID'].append(road_id)
        inputs['SHRP_ID'].append(road_name)
        for key, value in zip(headers[2:], row[1:]):
            if key in class_map.keys():
                inputs[key].append(class_map[key][value])
            elif key == headers[2]:
                inputs[key].append(int(value))
            elif key == 'PCI':
                inputs[key].append(float(value) if value != '' else None)
            else:
                inputs[key].append(float(value) if value != '' else -1)

print('修补PCI值：')
# 以下主要是处理缺失值
if data_type == 'train':
    data = {
        'batch': inputs['SHRP_ID'],
        'PCI': inputs['PCI'],
        'Construction': inputs['REAL_Construction_Type']
    }
    for idx, val in enumerate(tqdm(data['Construction'])):
        if val != 0 and data['PCI'][idx] is None:
            if val == 1:
                if data['batch'][idx-1] == data['batch'][idx] and data['PCI'][idx-1] is not None:
                    data['PCI'][idx] = min(100, max(0, -68.13*math.log(data['PCI'][idx-1])+314.04))
                else:
                    data['PCI'][idx] = 85
            elif val ==2 or val == 3:
                data['PCI'][idx] = 100
            elif val == 4:
                if data['batch'][idx-1] == data['batch'][idx] and data['PCI'][idx-1] is not None:
                    data['PCI'][idx] = min(100, max(0,-80.59*math.log(data['PCI'][idx-1])+371.24))
                else:
                    data['PCI'][idx] = 85
            elif val == 5:
                data['PCI'][idx] = 85
            elif val == 6:
                if data['batch'][idx-1] == data['batch'][idx] and data['PCI'][idx-1] is not None:
                    data['PCI'][idx] = min(100, max(0, -77.8*math.log(data['PCI'][idx-1])+358))
                else:
                    data['PCI'][idx] = 85
    df = pd.DataFrame(data)
    filled_df = df.groupby('batch', group_keys=False).apply(fill_pci_linear)
    inputs['PCI'] = filled_df['PCI'].tolist()

# 精确值转区间
if Filled_interval:
    # inputs['PCI'] = [get_interval_number(x) if x is not None else None for x in inputs['PCI']]
    inputs['PCI'] = [round(x) if x is not None else None for x in inputs['PCI']]
else:
    inputs['PCI'] = filled_df['PCI'].tolist()

print('正在扩充数据集，时间较长，耐心等待...')
df = pd.DataFrame(inputs)
df_newroad = new_road(df)
inputs = full_fill_data(df_newroad).to_dict(orient='list')

# 将所有处理好的数据保存为json文件，方便后续模型读取
with open(json_output, "w", encoding="utf-8") as f:
    json.dump(inputs, f, indent=4)
