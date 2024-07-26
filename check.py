import os
import torch
from PIL import Image
import argparse
import model
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/LeNet.pth')
    parser.add_argument('--img', type=str, default='None')
    parser.add_argument('--img-dir', type=str, default='./test')
    parser.add_argument('--view', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parse()
    Net = model.LeNet()
    try:
        missing_keys, unexpected_keys = Net.load_state_dict(torch.load(args.model))
    except FileNotFoundError:
        assert 'Model does not exist OR Model is destroyrd'
    f = open(os.path.join(args.img_dir, 'ans.txt'), 'w')  # 刷新文件，清空原有的数据
    f.close()
    Net.eval()
    files = filter(lambda x: x.endswith('.png'), os.listdir(args.img_dir))
    for img_name in files:
        img = Image.open(os.path.join(args.img_dir, img_name)).convert('RGB')
        img = transform(img).unsqueeze(0)  # 在前面增加一维，用来模拟batch（因为训练时前面有一维，训练模型也加了一维）
        output = Net(img)
        output = output.argmax()
        with open(os.path.join(args.img_dir, 'ans.txt'), 'a') as f:
            print(img_name + '->' + str(output.item()) + '\n')
            if args.view:
                f.write(img_name + '->' + str(output.item()) + '\n')
    print('The result is saved at {}'.format(os.path.join(args.img_dir, 'ans.txt')))
