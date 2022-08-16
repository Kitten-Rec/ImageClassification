import tensorflow as tf
from layers.SEnet import SE_Block

"""ref:
1. https://blog.csdn.net/EasonCcc/article/details/108649071?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-108649071-blog-114100031.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-108649071-blog-114100031.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=2
2. https://github.com/FeiYee/SE-ResNeXt-TensorFlow/blob/4b94e0439269a965e248502db186d0aaaa7d9c7a/SE_ResNeXt.py#L88
3. https://blog.csdn.net/dgvv4/article/details/123572065
4. https://wanghao.blog.csdn.net/article/details/117431319
"""

from layers.GroupConv import GroupConv2D


class SE_ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(SE_ResNeXt_BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters,
                                      output_channels=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,  # 只在分组卷积和跳连接的时候宽高发生变化
                                      padding="same",
                                      groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters,  # 输出通道数变为原来的2倍
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        # 加入SE_Block
        self.se = SE_Block(filter_num=2 * filters, reduction=16)
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters,  # 输出通道数变为原来的2倍 块输出通道数=块输入通道数
                                                    kernel_size=(1, 1),
                                                    strides=strides,  # 只在分组卷积和跳连接的时候宽高发生变化
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        # 通道注意力  插入SE_Block
        x = self.se(x)
        # x = tf.keras.layers.multiply([x, coefficient])

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def build_SE_ResNeXt_block(filters, strides, groups, repeat_num):
    block = tf.keras.Sequential()
    block.add(SE_ResNeXt_BottleNeck(filters=filters,
                                    strides=strides,  # 控制宽高缩小
                                    groups=groups))
    for _ in range(1, repeat_num):
        block.add(SE_ResNeXt_BottleNeck(filters=filters,
                                        strides=1,  # ??? 控制宽高不变
                                        groups=groups))

    return block


class SE_ResNeXt(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality, classes):
        if len(repeat_num_list) != 4:
            raise ValueError("The length of repeat_num_list must be four.")
        super(SE_ResNeXt, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.block1 = build_SE_ResNeXt_block(filters=128,
                                             strides=1,  # 宽高不缩小
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[0])
        self.block2 = build_SE_ResNeXt_block(filters=256,
                                             strides=2,  # 宽高缩小一半
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[1])
        self.block3 = build_SE_ResNeXt_block(filters=512,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[2])
        self.block4 = build_SE_ResNeXt_block(filters=1024,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[3])
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=classes,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x


def SE_ResNeXt50(classes):
    return SE_ResNeXt(repeat_num_list=[3, 4, 6, 3],
                      cardinality=32, classes=classes)


def SE_ResNeXt101(classes):
    return SE_ResNeXt(repeat_num_list=[3, 4, 23, 3],
                      cardinality=32, classes=classes)

# if __name__ == '__main__':
#     model = SE_ResNeXt50(classes=5)
#     model.built(input_shape=(8, 224, 224, 3))
#     model.summary()