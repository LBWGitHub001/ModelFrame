# 一个简单的神经网络框架
## 配置环境
### 依赖包
>1. `cd [path of the root of the Project]`  
>2. `pip install -r requirements.txt`  
>3. 安装torch和torchvision，注意要和你的cuda版本对应
### 安装cuda[Linux-Ubuntu]
cuda是Nvidia开发的帮助开发人员更高的利用GPU进行并行计算的框架，对于深度学习至关重要，可以起到加速模型的训练和推理
>1. 检查本机的显卡驱动是否安装好`nvidia-smi`如果显示信息说明已经安装完成  
>2. 首先进入[Nvidia Cuda官网](https://developer.nvidia.com/cuda-toolkit)下载安装包这里推荐下载
cuda的11.8版本，推荐runfile模式安装  
  1 `wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run`
  2 `sudo sh cuda_11.8.0_520.61.05_linux.run`  
>3. 安装完成后不要关闭终端，按照终端上的提示将库写入环境变量~/.bashrc
?4. 检查是否安装成功 `nvcc -V` 如出现编译器版本号则说明成功
### 安装cudnn
cuDNN是Nvidia开发的模型推理加速的拓展库，对于模型的部署至关重要
>1. 前往[Nvidia cuDNN官网](https://developer.nvidia.com/cudnn)下载最新版本即可  
>2. 注意此时不需要再次安装cuda，因此下面的两条命令不需要执行
### 安装Anaconda3
Anaconda是一个Python虚拟环境管理器，其中内置了大量Python模块，可以方便的管理模块和虚拟环境
>1. [Anaconda官网](https://www.anaconda.com/download/)下载，注意也要添加到环境变量中
### 安装PyTorch
PyTorch是现在流行的神经网络搭建架构，可以帮助开发人员快速实现神经网络的搭建
>1. 从[官网下载库](https://download.pytorch.org/whl/torch_stable.html)中下载，与cuda版本对应，如果你的cuda版本是11.8，python是3.8，那么应该安装 **[torch-2.0.0+cu118-cp38-cp38-linux_x86_64.whl]**
>2. 下载torchvision，同样版本要对应，不再赘述
## 数据集准备
### LeNet模型(以MNIST数据集为例)
文件夹结构
> - MNIST
>> - train
>>> - 【Floders】文件夹的名称就是标签
>> - val
>>> - 【Floders】文件夹的名称就是标签
>> - labels.yaml( **以yaml语法标出类名和id的对应，id从0开始，需连续** )

## 训练
1. `python train.py`
### 可选参数
>[--epochs] 训练的扫描数据集的次数  
>[--batch-size] 数据集批量大小  
>[--lr] 学习率大小
## 测试
1. `python check.py`
### 可选参数
>[--model] 模型路径  
>[--img] 单张图像  
>[--img-dir] 一个目录的图像批量测试
>[--view] 是否要保存测试结果
## 模型转化
> 详细见子仓库export