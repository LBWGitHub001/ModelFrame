import numpy as np
from matplotlib import pyplot as plt
class DrawGraph:
    def __init__(self):
        # 创建实时绘制横纵轴变量
        self.it=0
        self.x = []
        self.y = []
        plt.ion()

    def add(self, y):
        self.x.append(self.it)
        self.y.append(y)
        self.it+=1
        self.draw()
        return self

    def draw(self):
        plt.ioff()  # 关闭画图窗口
        plt.clf()  # 清除之前画的图
        plt.plot(self.x, self.y)  # 画出当前x列表和y列表中的值的图形

    def save(self, filename):
        self.draw()
        plt.savefig(filename)

