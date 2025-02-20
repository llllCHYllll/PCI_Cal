import json
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

features = ['survey_date', 'REAL_SS_MATL', 'GS_THICKNESS', 'REAL2_GS_MATL', 'TS_THICKNESS', 'REAL_TS_MATL', 
           'GB_THICKNESS', 'REAL_GB_MATL', 'TB_THICKNESS', 'REAL_TB_MATL', 'PC_THICKNESS', 'AC_THICKNESS', 
           'REAL_AC_MATL', 'REAL_Construction_Type', 'LANE_WIDTH', 'LANE_LENGTH', 'LANES_NO', 'PRECIPITATION', 
           'EVAPORATION', 'EMISSIVITY_AVG', 'SHORTWAVE_SURFACE_AVG', 'REL_HUM_AVG_AVG', 'TEMP_AVG', 
           'DAYS_ABOVE_32_C', 'DAYS_BELOW_0_C', 'FREEZE_INDEX', 'FREEZE_THAW', 'AADT_ALL_VEHIC_2WAY', 
           'AADT_TRUCK_COMBO_2WAY', 'AADT_ALL_VEHIC', 'AADT_TRUCK_COMBO', 'ANL_KESAL_LTPP_LN_YR', 'prev_PCI']
target = 'PCI'
train_data = './rawdata/filled_interval_all.json'

batch_size = 128

init_dim = len(features)

with open(train_data, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df = pd.DataFrame(data)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class IntervalModel(nn.Module):
    def  __init__ (self):
        super(). __init__ ()
        self.net = nn.Sequential(
            nn.Linear(init_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.net(x)

class CustomLoss(nn.Module):
    def  __init__ (self):
        super(). __init__ ()
        
    def forward(self, pred, target, prev_pci, pv_values):
        lower = pred[:, 0].unsqueeze(1)
        upper = pred[:, 1].unsqueeze(1)
        
        # 基础损失项
        center_loss = ((target - (lower + upper)/2)**2).mean()  # 中心点误差
        width_loss = (upper - lower).mean()                     # 区间宽度
        valid_penalty = torch.clamp(lower - upper, min=0).mean()# 保证上界>=下界
        
        # 先验知识惩罚项
        direction_penalty = 0
        for i in range(len(pv_values)):
            current_pv = pv_values[i]
            prev_e = prev_pci[i]
            center = (lower[i] + upper[i])/2
            
            if current_pv == 0:
                # 当pv=0时，预测中心应<=前值
                direction_penalty += torch.relu(center - prev_e)
            else:
                # 当pv≠0时，预测中心应>=前值
                direction_penalty += torch.relu(prev_e - center)
        
        # 总损失组合
        total_loss = (
            center_loss * 10
            + 0.1 * width_loss 
            + 10 * valid_penalty 
            + 0.5 * direction_penalty/len(pv_values)
        )
        return total_loss

model = IntervalModel()
criterion = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # lr=0.001
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

epochs = 200
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:

        pv_values = batch_X[:, 13]  
        prev_pci = batch_X[:,-1]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y, prev_pci, pv_values)
        loss.backward()
        optimizer.step()
        # scheduler.step()
    
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    predictions = model(X)
    lower = predictions[:, 0].numpy()
    upper = predictions[:, 1].numpy()

print("\n预测结果（包含先验知识约束）：")
for i in range(len(df)):
    current_e = df['PCI'][i]
    current_pv = df['REAL_Construction_Type'][i]
    trend = "▲" if current_pv !=0 else "▼"
    print(f"样本{i}: {current_e:.1f} → {trend} 预测范围: [{lower[i]:.2f}, {upper[i]:.2f}]")
