from tensorflow import keras
from tensorflow.keras import layers, Sequential
from layers.SKnet import SKConv


class SKBlock(layers.Layer):

    def __init__(self, filter_num, flag, stride):
        super(SKBlock, self).__init__()
        # self.conv1 = layers.Conv2D(filter_num, (1, 1), strides=1, kernel_initializer='he_normal', padding='same')
        self.conv1 = layers.Conv2D(filter_num, (1, 1), strides=stride, kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization(axis=3)
        self.relu1 = layers.Activation('relu')

        # 与 ResNet block最大的区别就在于中间的这个3*3的卷积 使用 SkConv进行取代
        self.conv2 = SKConv(in_channels=filter_num, out_channels=filter_num, kernel_size=3, stride=1,
                            kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization(axis=3)
        self.relu2 = layers.Activation('relu')

        self.conv3 = layers.Conv2D(filter_num * 4, (1, 1), strides=1, kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization(axis=3)

        if flag == 2:  # 只有第一组的第一个block 长宽size不变 通道数*4  其他都是size*2 channel*2
            self.downsample = Sequential([
                layers.Conv2D(filter_num * 4, (1, 1), strides=stride, kernel_initializer='he_normal'),
                layers.BatchNormalization(axis=3)
            ])
        else:  # flag == 1 恒等变换
            self.downsample = lambda x: x

        self.relu3 = layers.Activation('relu')

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = self.relu3(output)
        return output


def build_resblock(filter_num, blocks, stride):
    res_blocks = Sequential()
    res_blocks.add(SKBlock(filter_num, flag=2, stride=stride))  # flag=2 : 每组的第一个block的Identity需要变换通道数
    # stride : 每组的第一个block控制长宽尺寸的变化
    for _ in range(1, blocks):
        res_blocks.add(SKBlock(filter_num, flag=1, stride=1))
    return res_blocks


class SK_ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes):
        super(SK_ResNet, self).__init__()
        self.stem = Sequential([
            layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad'),  # 需要额外进行padding
            layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_normal', padding="valid"),
            layers.BatchNormalization(axis=3),
            layers.Activation('relu'),
            layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
        ])
        # layer1 长宽尺寸不变  layer2~4 长宽尺寸将为原来的一半
        self.layer1 = build_resblock(64, layer_dims[0], stride=1)
        self.layer2 = build_resblock(128, layer_dims[1], stride=2)  # 这里的stride是为了identity是否需要变换维度，1为不需要变换，2需要
        self.layer3 = build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def SK_ResNet50(classes):
    net = SK_ResNet([3, 4, 6, 3], num_classes=classes)
    return net


def SK_ResNet101(classes):
    net = SK_ResNet([3, 4, 23, 3], num_classes=classes)
    return net


def SK_ResNet152(classes):
    net = SK_ResNet([3, 8, 36, 3], num_classes=classes)
    return net


# if __name__ == '__main__':
#     batch_size = 8
#     model = SK_ResNet50(classes=5)
#     model.build(input_shape=(batch_size, 224, 224, 3))
#     model.summary()  # 打印模型网络结构
