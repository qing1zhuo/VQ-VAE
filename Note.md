忍无可忍遂决定写一写日记方便记录

# 1 MNIST 数据集实验

## 1.1 VQ-VAE重建实验
#### 参数配置
- 训练参数
EPOCHS=15
PRINT_EPOC=1
BATCH_SIZE=128  # 原论文128
lr=2e-4
wd=0
device="cuda:0" if torch.cuda.is_available() else "cpu"
- 模型参数
IN_CHANNELS=1
HIDDEN_CHANNELS=128
OUT_CHANNELS=1
RESIDUAL_CHANNELS=32
RESIDUAL_LAYERS=2
NUM_EMBEDDINGS=15
EMBEDDING_DIM=64
DECAY=0.99
COMMIT_COST=0.25
- 数据预处理
resize(32)
mean=[0.0]
std=[1.0]
#### 训练损失
![alt text](.\picture\MNIST\vaeloss.png)
最低总损失：0.003365
![alt text](.\picture\MNIST\perplexity.png)
最高码本利用率：7.5/15
#### 重建效果
1. train
![alt text](.\picture\MNIST\recon_train.png)
2. test
![alt text](.\picture\MNIST\recon_test.png)
分析：基本可以达到将近无损重建

## 1.2 PixelCNN先验自回归模型训练
#### 参数配置
- 训练参数
EPOCHS=10*4
PRINT_EPOC=1
BATCH_SIZE=128
lr=2e-4
wd=0
- 模型参数
INPUT_DIM=15
DIM=64
LAYERS=15
N_CLASS=10
INDICES_H = 8
INDICES_W = 8
- 数据预处理
mean=[0.0]
std=[1.0]
#### 训练损失
![alt text](.\picture\MNIST\pixelcnn_loss.png)
测试集最低损失：0.951427
#### 采样生成
![alt text](.\picture\MNIST\generate.png)

## 1.3 分析
对于VQ-VAE训练，将原数据resize为32*32
整体的重建和生成效果都很好，我认为原因在于类别少、图像简单、样本较标签而言比较丰富

# 2 CIFAR10 数据集
## 2.1 VQ-VAE训练
#### 参数配置
- 训练配置
EPOCHS=100
PRINT_EPOC=10
BATCH_SIZE=256
lr=2e-4
wd=0
device="cuda:0" if torch.cuda.is_available() else "cpu"
- 模型配置
IN_CHANNELS=3
HIDDEN_CHANNELS=128
OUT_CHANNELS=3
RESIDUAL_CHANNELS=32
RESIDUAL_LAYERS=4
NUM_EMBEDDINGS=512    
EMBEDDING_DIM=128
DECAY=0.99
COMMIT_COST=0.25
- 数据预处理
mean=[0.4914, 0.4822, 0.4465]
std=[0.2470, 0.2435, 0.2616]
#### 训练损失
![alt text](.\picture\CIFAR\vaeloss.png)
最低总损失：0.091955
![alt text](.\picture\CIFAR\perplexity.png)
最高码本利用率：402/512
#### 重建效果
1. train
![alt text](.\picture\CIFAR\recon_train.png)
2. test
![alt text](.\picture\CIFAR\recon_test.png)

## 2.2 PixelCNN训练
#### 参数配置
- 训练参数
EPOCHS=10
PRINT_EPOC=1
BATCH_SIZE=128
lr=2e-4/1e-4    
wd=1e-5/3e-5
device="cuda:0" if torch.cuda.is_available() else "cpu"
- 模型参数
INPUT_DIM=512
DIM=128
LAYERS=20
N_CLASS=10
INDICES_H = 8
INDICES_W = 8
- 数据预处理
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
#### 训练损失
1. 100轮：lr=2e-4/wd=1e-5
![alt text](.\picture\CIFAR\pixelcnn_loss1.png)
最低验证集损失：4.758106
2. 10轮：lr=2e-4/wd=3e-5
![alt text](.\picture\CIFAR\pixelcnn_loss2.png)
最低验证集损失：4.535129
3. 10轮：lr=1e-4/wd=3e-5
![alt text](.\picture\CIFAR\pixelcnn_loss3.png)
最低验证集损失：4.461217
#### 采样生成
![alt text](.\picture\CIFAR\generate.png)
![alt text](.\picture\CIFAR\generate2.png)

## 2.3 分析
重建图片能够基本辨认，相较原图较为模糊，我认为原因在于原始数据集分辨率就不高
生成效果很一般，能大概看出物体的轮廓，但几乎没有具体细节。
生成效果一般的原因：
1. 原始图像的分辨率本身就不高；
2. 特征索引图为$8*8*1$，压缩率为$\frac{32*32*3}{8*8*1}=48$，信息损失本身较多
3. 图片本身内容特征更加丰富，采取和MNIST同样的特征图大小，重建更为困难
4. 码本数量更多，模型要从更多的码本向量里面挑选
生成效果尚且有个轮廓的原因：
1. 每个类别有6000张，足够模型学到较为基础的轮廓、颜色特征


# 3. Mini-ImageNet 数据集 (MIN)
## 3.1 VQ-VAE训练
#### 参数配置
- 训练配置
EPOCHS=20
PRINT_EPOC=1
BATCH_SIZE=256
lr=2e-4
wd=0
device="cuda:0" if torch.cuda.is_available() else "cpu"
- 模型配置
IN_CHANNELS=3
HIDDEN_CHANNELS=128
OUT_CHANNELS=3
RESIDUAL_CHANNELS=32
RESIDUAL_LAYERS=4
NUM_EMBEDDINGS=512    
EMBEDDING_DIM=128
DECAY=0.99
COMMIT_COST=0.25
- 数据预处理
resize(128)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
#### 训练损失
![alt text](.\picture\MIN\vqvaeloss.png)
最低总损失：0.065451
![alt text](.\picture\MIN\perplexity.png)
最高码本利用率：339/512
#### 重建效果
1. train
![alt text](.\picture\MIN\recon_train.png)
2. test
![alt text](.\picture\MIN\recon_test1.png)
![alt text](.\picture\MIN\recon_test2.png)

## 3.2 PixelCNN训练
#### 参数配置
- 训练参数
EPOCHS=30+10
PRINT_EPOC=1
BATCH_SIZE=32
lr=2e-4
wd=1e-5
device="cuda:0" if torch.cuda.is_available() else "cpu"
- vqvae模型配置
IN_CHANNELS=3
HIDDEN_CHANNELS=128
OUT_CHANNELS=3
RESIDUAL_CHANNELS=32
RESIDUAL_LAYERS=2
NUM_EMBEDDINGS=512    
EMBEDDING_DIM=128
DECAY=0.99
COMMIT_COST=0.25
- pixelcnn模型配置
INPUT_DIM=512
DIM=128
LAYERS=20
N_CLASS=100
INDICES_H = 32
INDICES_W = 32
- 数据预处理
resize(128)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
#### 训练损失
1. 30轮
![alt text](.\picture\MIN\pixelcnn_loss.png)
最低验证集损失：3.200977
#### 采样生成
![alt text](.\picture\MIN\generate.png)
#### 码本使用率