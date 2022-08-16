import tensorflow as tf
from tensorflow.keras import layers, Sequential
from layers.GroupConv import GroupConv2D
from functools import reduce
from layers import GlobalAveragePooling2D

"""ref： 
    https://blog.csdn.net/practical_sharp/article/details/115032956
    https://github.com/rwightman/pytorch-image-models/blob/a7f95818e44b281137503bcf4b3e3e94d8ffa52f/timm/models/sknet.py#L165
"""


class SKConv(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, M=2, r=16, L=32, kernel_initializer='he_normal'):
        """
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        """
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到 向量Z 的长度d
        # print("d:  ", d)
        self.M = M
        self.out_channels = out_channels
        self.conv = []
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(Sequential([
                layers.ZeroPadding2D((1 + i, 1 + i)),
                # 组卷积+空洞卷积
                GroupConv2D(in_channels, out_channels, kernel_size=kernel_size, strides=(stride, stride),
                            dilation_rate=(1 + i, 1 + i), groups=32, use_bias=False, kernel_initializer=kernel_initializer),
                layers.BatchNormalization(),
                layers.ReLU()
            ]))
        self.global_pool = GlobalAveragePooling2D()  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = Sequential([
            layers.Conv2D(d, kernel_size=1, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization(),
            layers.ReLU()
        ])  # 降维
        self.fc2 = layers.Conv2D(out_channels * M, kernel_size=1, strides=(1, 1), use_bias=False, kernel_initializer=kernel_initializer)  # 升维
        self.softmax = layers.Softmax(axis=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b=1

    def call(self, inputs, **kwargs):
        # batch_size = inputs.shape[0]
        outputs = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            outputs.append(conv(inputs))  # [batch_size,H,W,out_channels]
            # the part of fusion
        U = reduce(lambda x, y: x + y, outputs)  # 逐元素相加生成 混合特征U  [batch_size,H,W,channel]
        # print("U.size():   ", U.shape)
        s = self.global_pool(U)  # [batch_size,1,1,channel]
        # tf.reshape(s, [-1, s.shape[1], 1, 1])
        # print("s.size():   ", s.shape)
        z = self.fc1(s)  # S->Z降维   # [batch_size,1,1,d]
        # print("z.size():   ", z.shape)
        a_b = self.fc2(z)  # Z->a_b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,1,1,out_channels*M]
        # print("a_b.size():  ", a_b.shape)
        a_b = tf.reshape(a_b, [-1, self.M, 1, self.out_channels])  # 调整形状，变为 两个全连接层的值[batch_size,M,1,out_channels]
        # print("a_b.size():  ", a_b.shape)
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        # the part of selection
        # tf.split(a_b, [1]*self.M, axis=1)
        a_b = tf.keras.layers.Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': self.M})(a_b)
        # print(len(a_b))  # a_b:[[batch_size,1,1,out_channels],[batch_size,1,1,out_channels]]
        # print("a_b[0].size():    ", a_b[0].shape)
        # print("a_b[1].size():    ", a_b[1].shape)

        V = list(map(lambda x, y: x * y, outputs, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]

        V = reduce(lambda x, y: x + y, V)  # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        # print("V.size():   ", V.shape)
        return V  # [batch_size,H,W,out_channels]

#
# if __name__ == '__main__':
#     x = layers.Input(shape=(24, 24, 2048))
#     # x = torch.Tensor(8, 2048, 24, 24)
#     SKConv(in_channels=2048, out_channels=4096, stride=1, M=2, r=16, L=32)(x)

