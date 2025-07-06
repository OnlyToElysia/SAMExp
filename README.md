# 新型优化算法（SAM）的实现与分析 



## 1.实验环境

本次实验所使用的环境如下：
| 条目       | 所用环境       | 
|------------|----------------|
| 操作系统    | Windows 11        | 
| CPU        | AMD Ryzen 7 7735H |
| 内存        | 8G          | 
| GPU        | NVIDIA RTX 4060(Laptop,8G) | 
| CUDA版本    | 12.8            | 
| Python版本  | 3.10           | 

python环境需要安装如下依赖：

```
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  
numpy>=1.24.0         
matplotlib>=3.7.1     
tqdm>=4.65.0
numpy
```

## 2.数据集下载

本次实验所使用的的数据集CIFAR10可在代码运行时自动下载。具体来说，由以下语句控制：

```
train_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=False, download=True, transform=transform_test)
```

## 3.运行方式

输入如下指令以运行程序：

```
python SAM_Resnet18.py
python SAM_EfficientNet.py
Ablation_Experiments.py
```

## 4.实验结果

本次实验以各方法训练100轮后的测试集准确率为评估标准之一，以下是结果：

| 条目       | 结果       | 
|------------|----------------|
| SGD    | 87.77%         | 
| Adam        | 78.44%  |
| SAM        | 88.29%         | 
| ImprovedSAM | 88.50% | 

由结果可得，SAM相较于SGD和Adam等传统方法有较为明显的优势，而优化过后的SAM又在此基础上有了提升。
其它相关的图类结果及相关分析请见课程设计报告。
