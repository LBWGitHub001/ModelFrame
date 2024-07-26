import torch
import torch.onnx
from model import LeNet
from conf import *
import os

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = LeNet()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    checkpoint = './export/LeNet.pth'
    onnx_path = './export/LeNet.onnx'
    input = torch.randn(1, 3, 32, 32)  #这里用的是自己的模型，数据需要根据自己的模型进行更改。
    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)
