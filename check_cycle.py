import torch
import torch.nn as nn
import numpy as np

class JetRulModel(nn.Module):
    def __init__(self,no_of_features):
        super(JetRulModel, self).__init__()
        self.linear1 = nn.Linear(no_of_features,20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20,1)
        
    def forward(self, targets_train ):
        out = self.linear1(targets_train)
        out  = self.relu(out)
        out = self.linear2(out)
    
        return out

model = JetRulModel(no_of_features=10)
model.load_state_dict(torch.load('weights.pth'))

def predict_value(inputs):
    if len(inputs)==10:
        inputs = np.array(inputs)/100
        inputs = torch.from_numpy(inputs.astype(np.float32))
        pred = model(inputs)
        value = pred.item() * 100
        return int(value)
    
    return "Input Error!"