import os.path
import torch
import yaml
import argparse
from drawGraph import *
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from DataSet import MyDataset
from model import LeNet

# %%参数设置
# batch_size = 256
# lr = 0.03
# epochs = 5

# %%数据集准备
train_dir = './MNIST/train'
val_dir = './MNIST/val'
yaml_path = './MNIST/labels.yaml'

transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

with open(yaml_path, 'r') as file:
    labels = yaml.safe_load(file)


class ModelMannager():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.train_data = MyDataset(train_dir, transform=transforms, label_name=labels)

        self.val_data = MyDataset(val_dir, transform=transforms, label_name=labels)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        # %%初始化模型
        # 训练设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:",self.device)

        # 网络
        self.Net = LeNet().to(self.device)
        self.Net.initialize_weights()

        # 损失函数
        self.Loss = nn.CrossEntropyLoss()

        # 更新器
        self.Updater = torch.optim.SGD(self.Net.parameters(), lr=self.lr, momentum=0.9)

        # 画图
        self.draw_loss = DrawGraph()
        self.draw_accuracy = DrawGraph()

    # %%训练和验证正确率

    # 验证
    def val(self):
        rights = 0
        total = 0
        self.Net.eval()
        with torch.no_grad():
            for img, label in self.val_loader:
                img, label = img.to(self.device), label.to(self.device)  # 数据迁移到训练设备
                outputs = self.Net(img)

                _, predicted = torch.max(outputs.data, 1)
                # 更新计数器
                total += label.size(0)
                rights += (predicted == label).sum().item()
        return rights / total

    # 训练
    def train(self):
        self.Net.train(True)
        for epoch in range(self.epochs):
            self.Net.train(True)
            batch_all = len(self.train_data)
            batch_count = 0
            for img, label in self.train_loader:
                img, label = img.to(self.device), label.to(self.device)  # 数据迁移到训练设备
                self.Updater.zero_grad()  # 清空过往梯度
                output = self.Net(img)  # 模型推理
                loss = self.Loss(output, label)  # 损失计算
                loss.backward()  # 向后传播，计算梯度
                self.Updater.step()  # 参数更新
                if batch_count % 50 == 0:
                    print('Epoch [{}/{}][{}], Loss: {:.4f}'.format(epoch + 1, self.epochs, 300 * batch_count / batch_all,
                                                                   loss.item()))
                batch_count += 1
                self.draw_loss.add(loss.item())
            accuracy = self.val()
            self.draw_accuracy.add(accuracy)
            print('Validation Accuracy: {:.4f} Epoch[{}/{}]'.format(accuracy, epoch + 1, self.epochs))

    def SaveGraph(self, loss_path, accuracy_path):
        self.draw_loss.save(loss_path)
        self.draw_accuracy.save(accuracy_path)

    def StartTrain(self):
        self.train()

    def Save(self,save_path,postfix='.pth'):
        mode_path = save_path + postfix
        torch.save(self.Net.state_dict(), mode_path)


def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--model", type=str, default='LeNet')
    parser.add_argument("--save-path", type=str, default='./models')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    MM = ModelMannager(args)
    MM.StartTrain()
    MM.SaveGraph(loss_path="./result/loss.png", accuracy_path="./result/accuracy.png")
    MM.Save(os.path.join(args.save_path, args.model))