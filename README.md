## 复现论文题目
A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction

## 数据链接
(https://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)

## 文件夹介绍
文件夹data下存放的是数据(**运行程序后会自动生成**);

文件夹log下存放每一次运行保存的模型(**运行程序后会自动生成**);

文件夹logs_xxxxx为保存的tensorboard(通过日期指定名称);

文件夹model下为3个.py文件,分别为:
- DSARNN.py  --**模型**
- lookahead_optimizer.py  --优化算法
- randam_optimizer.py   --优化算法

文件夹paper下为论文.pdf

文件夹 ulti下为读取数据的.py,分别为:
- feature_extract.py  -- 提取特征,程序中没有用到该.py
- get_data_train.py   -- 得到训练集和验证集,**测试集没有被用到**
config.py为配置程序

**main_test_darnn.py为主程序**       
## 程序执行需求
程序需要安装以下库才能运行:
- tensorflow-gpu == 1.10.0
- h5py
- tqdm

## 程序执行
python main_test_darnn.py
程序执行后,会在log中保存模型和保存logs_xxxxxx,可以通过tensorboad --logdir logs_xxxxxx查看权重,
偏置,损失函数等的收敛情况.

### 关于文章的简单介绍,可以看我的[博客](https://www.jianshu.com/p/cb9767ce73f0),欢迎交流留言 ~~
