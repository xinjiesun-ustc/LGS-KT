from torch.utils.data import DataLoader
from data_load_codenet_python import read_file


import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1025)  # 你可以选择你想要的任何种子值

NUM_PROBLEMS = 81492 #codenet-python  用户369个  选择大于100条做题记录   平均长度为 220.8455
NUM_QUESTIONS = 1493  #codenet-python
EMBED_SIZE = 256
BATCH_SIZE = 32

# gramerscore最大值： 2085.0
# gramerscore最小值： 0
# cpu_time： 42060
# cpu_time： 0
# memory： 7796236
# memory： 0
num_g = 2085
num_cpu = 42060
num_mem= 7796236  #这个特别占用资源 导致程序变慢的主要原因
#读取数据
train_students = read_file("./data/Afterprocess/train_codenet_python_xz.txt")
test_students = read_file("./data/Afterprocess/test_codenet_python_xz.txt")


#按batch加载数据
train_data_loader = DataLoader(train_students, batch_size=BATCH_SIZE, shuffle=True)  # 创建数据加载器
test_data_loader = DataLoader(test_students, batch_size=BATCH_SIZE, shuffle=True)  # 创建数据加载器
from DKT_emb_codenet_python import myKT_DKT
dkt = myKT_DKT(NUM_PROBLEMS,NUM_QUESTIONS,num_g,num_cpu,num_mem,EMBED_SIZE)
dkt.train(train_data_loader, test_data_loader, epoch=10)
dkt.save("dkt.params")
dkt.load("dkt.params")
auc,acc = dkt.eval(test_data_loader)
print("auc: %.6f" % auc)