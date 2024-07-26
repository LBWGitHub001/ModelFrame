# 模型格式转换
## 目前仅支持pth,onnx,pb,kmodel格式之间的转换
>运行  
> `pip install -r requirements.txt`  
> 完成依赖安装，所需要的TensorFlow环境需要根据你的电脑配置进行安装，故不在此说明

>1. pth->onnx  
  需要PyTorch环境，运行pth2onnx.py即可完成转换，注意需要修改路径
>2. onnx->pb  
  需要TensorFlow环境，运行onnx2pb即可完成转换