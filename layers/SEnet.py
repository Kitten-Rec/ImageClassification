from tensorflow.keras import layers


class SE_Block(layers.Layer):
    def __init__(self, filter_num, reduction=16):
        super(SE_Block, self).__init__()
        self.filter_num = filter_num
        self.reduction = reduction
        # se-block
        self.se_globalpool = layers.GlobalAveragePooling2D()
        self.se_resize = layers.Reshape((1, 1, self.filter_num))
        self.se_fc1 = layers.Dense(units=self.filter_num // self.reduction, activation='relu', use_bias=False)
        self.se_fc2 = layers.Dense(units=self.filter_num, activation='sigmoid', use_bias=False)

    def call(self, inputs, **kwargs):
        x = self.se_globalpool(inputs)
        x = self.se_resize(x)
        x = self.se_fc1(x)
        x = self.se_fc2(x)
        outputs = x * inputs
        # outputs += inputs  # 残差连接  这应该是错误的 https://www.jianshu.com/p/9faec670d8e4
        return outputs

