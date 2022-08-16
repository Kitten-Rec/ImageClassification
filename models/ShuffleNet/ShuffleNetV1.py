import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model

from layers.GroupConv import GroupConv2D

""" ref: https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1
 推理速度慢： '在相同的FLOPs条件下，depth-wise卷积需要的IO读取次数是普通卷积的100倍' https://zhuanlan.zhihu.com/p/149564248
"""


class ShuffleV1Block(layers.Layer):
    def __init__(self, inp, oup, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw
            GroupConv2D(inp, mid_channels, kernel_size=1, strides=(1, 1), groups=1 if first_group else group,
                        use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # dw
            layers.ZeroPadding2D(padding=(pad, pad)),
            GroupConv2D(mid_channels, mid_channels, ksize, (stride, stride), groups=mid_channels, use_bias=False),
            layers.BatchNormalization()
        ]
        branch_main_2 = [
            # pw-linear
            GroupConv2D(mid_channels, outputs, kernel_size=1, strides=(1, 1), groups=group, use_bias=False),
            layers.BatchNormalization()
        ]
        self.branch_main_1 = Sequential(branch_main_1)
        self.branch_main_2 = Sequential(branch_main_2)

        if stride == 2:
            self.branch_proj = Sequential([
                layers.ZeroPadding2D(padding=(1, 1)),
                layers.AveragePooling2D(pool_size=(3, 3), strides=2)
            ])

    def call(self, inputs, **kwargs):
        x = inputs
        x_proj = inputs

        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return tf.nn.relu(x + x_proj)  # 能保证channel相同 因为每个stage除第一个block外(stride == 1) in_channel = out_channel
        elif self.stride == 2:
            # concate(channel_in, channel_output) ==> inp + output = intp + (oup - inp) = oup
            return tf.concat([self.branch_proj(x_proj), tf.nn.relu(x)], -1)  # (B,  H, W, C)

    def channel_shuffle(self, x):
        batchsize, height, width, num_channels = x.shape
        # print(batchsize, height, width, num_channels)
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group  # 论文中的n

        # (B,  H, W, C) ==> (B, H, W, n, g)
        x = tf.reshape(x, [batchsize, height, width, group_channels, self.group])
        x = layers.Permute((1, 2, 4, 3))(x)  # (B, H, W, n, g) ==> (B, H, W, g, n) shuffle操作
        x = tf.reshape(x, [batchsize, height, width, num_channels])

        return x


class ShuffleNetV1(Model):
    def __init__(self, classes, model_size='2.0x', group=None):
        super(ShuffleNetV1, self).__init__()
        assert group is not None

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]  # !!!第二个参数好像错了 应当是36
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer / stage
        input_channel = self.stage_out_channels[1]
        self.first_conv = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),
            layers.Conv2D(filters=input_channel, kernel_size=3, strides=2, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.maxpool = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),
            layers.MaxPool2D(pool_size=(3, 3), strides=2)
        ])

        self.features = []
        # 每个stage 2~4
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            # stage重复的次数
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1  # 每个stage的第一个block长宽减半
                first_group = idxstage == 0 and i == 0  # 在是 stage2的第一个block 的情况下 first_group为true
                # 论文中说 'Note that for Stage 2, we do not apply group convolution on the first pointwise layer because the number of input channels is relatively small.'
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                                    group=group, first_group=first_group,
                                                    mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel  # 每个stage的第一个block channel有变化 后面的block channel没有变化, 也就是说channel的变化在每个stage的第一个block就完成了

        self.features = Sequential(self.features)
        self.globalpool = layers.GlobalAveragePooling2D()  # (batch_size, channels)
        self.classifier = layers.Dense(units=classes, use_bias=False, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)

        x = self.globalpool(x)
        x = self.classifier(x)
        return x

# if __name__ == '__main__':
#     # x = tf.keras.Input(224, 224, 3)
#     model = ShuffleNet(group=3, classes=5)
#     model.build((None, 224, 224, 3))
#     model.summary()
