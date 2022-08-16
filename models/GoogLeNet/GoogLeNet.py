from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


def GoogLeNet(input_shape, classes, aux_logits=False):
    input = input_shape  # 224x224x3
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(input)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=1,  activation="relu")(x)
    x = layers.Conv2D(filters=192, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = Inception(64, 96, 128, 16, 32, 32, name="3a")(x)
    x = Inception(128, 128, 192, 32, 96, 64, name="3b")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = Inception(192, 96, 208, 16, 48, 64, name="4a")(x)
    if aux_logits:
        aux1 = InceptionAux(classes=classes, name="aux1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="4b")(x)
    x = Inception(128, 128, 256, 24, 64, 64, name="4c")(x)
    x = Inception(112, 144, 288, 32, 64, 64, name="4d")(x)
    if aux_logits:
        aux2 = InceptionAux(classes=classes, name="aux2")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="4e")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = Inception(256, 160, 320, 32, 128, 128, name="5a")(x)
    x = Inception(384, 192, 384, 48, 128, 128, name="5b")(x)
    x = layers.AveragePooling2D((7, 7), strides=1)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.4)(x)
    x = layers.Dense(units=classes)(x)
    aux3 = layers.Softmax()(x)

    if aux_logits:
        model = Model(inputs=input, outputs=[aux1, aux2, aux3])
    else:
        model = Model(inputs=input, outputs=aux3)
    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3redu, ch3x3, ch5x5redu, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, padding="same", activation="relu")
        self.branch2 = Sequential([
            layers.Conv2D(ch3x3redu, kernel_size=1, activation="relu"),
            layers.Conv2D(ch3x3, kernel_size=3, padding="same", activation="relu")
        ])
        self.branch3 = Sequential([
            layers.Conv2D(ch5x5redu, kernel_size=1, activation="relu"),
            layers.Conv2D(ch5x5, kernel_size=5, padding="same", activation="relu")
        ])
        self.branch4 = Sequential([
            layers.MaxPooling2D((3, 3), strides=1, padding="same"),
            layers.Conv2D(pool_proj, kernel_size=1, activation="relu")
        ])

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionAux(layers.Layer):
    def __init__(self, classes, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.avgpool = layers.AveragePooling2D((5, 5), strides=3, padding="same")
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")
        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(classes)
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.avgpool(inputs)
        x = self.conv(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x