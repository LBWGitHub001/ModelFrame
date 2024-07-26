import os
import torch
from torch.utils.data import *
from torchvision import datasets, transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data_dir, label_name, transform=None):
        self.label_name = label_name
        self.data_info = self.get_img_info(data_dir)  # 存储所有图片路径和标签
        self.transform = transform

    def __getitem__(self, index):  # 必须重写此方法，否则报错
        img_path, label = self.data_info[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)  # 调用transform方法将图片进行转化
        return img, label

    def __len__(self):
        return len(self.data_info)

    # @staticmethod
    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, files in os.walk(data_dir):
            # 访问子目录，也就是100和1目录
            for sub_dir in dirs:
                # path.join()的作用是合并目录结构
                img_paths = os.path.join(root, sub_dir)
                # 使用filter过滤出所有jpg文件
                img_names = list(filter(lambda x: x.endswith('.png'), os.listdir(img_paths)))
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    img_path = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((img_path, label))
        return data_info
