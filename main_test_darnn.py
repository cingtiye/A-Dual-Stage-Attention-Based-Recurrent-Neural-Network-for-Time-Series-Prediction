# -*- coding: utf-8 -*-

from model import DSARNN
from ulti import get_data_train
import time
import warnings

warnings.filterwarnings('ignore')

# 获取训练集,验证集
train_data, valid_data, _ = get_data_train.run()

## 初始化DSARNN类
dSARnn = DSARNN.Attention(INPUT_LENGTH = 81,     # 传感器数目
                          TIME_STEP    = 10,     # 窗口程度
                          ENCODE_CELL  = 128,    # 编码器隐层单元数目
                          DECODE_CELL  = 128,    # 解码器隐层单元数目
                          LEARN_RATE   = 0.001,  # 初始化学习率
                          BATCH_SIZE   = 128)    # 批次
## 打印关键tensor
dSARnn.print_tensor()
## 训练模型
dSARnn.train_model(train_x      = train_data[0],
                   train_hist_y = train_data[1],
                   train_y      = train_data[2],
                   valid_x      = valid_data[0],
                   valid_hist_y = valid_data[1],
                   valid_y      = valid_data[2],
                   batch_size   = 128,            # 批次
                   num_epochs   = 128*2,          # 迭代次数=num_epochs*(train_data[0].shape[0]/batch_size)
                   num_threads  = 8,              # 电脑核数
                   save_name    = time.strftime("%Y%m%d%H%M"), )  # log文件夹下模型的名称

