

import torch
import wandb
import numpy as np

wandb.init(project="debug")


datas = []
for i in range(10):
    datas.append(torch.rand((5,6))[:,0].numpy())
    
data_list = datas
print(type(data_list))
print(type(data_list[0]))
print(data_list[0])
#data_list = np.random.rand(100).tolist()

hist = wandb.Histogram(data_list)

data = {}

def addData(tag, new_data):
    global data
    data[tag] = new_data

addData("test_hist", wandb.Histogram(data_list))

# Log as a W&B Histogram object
wandb.log(data)

wandb.finish()



