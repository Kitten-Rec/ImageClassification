from tensorflow.keras import layers
from tensorflow.python.keras import backend


class GlobalAveragePooling2D(layers.Layer):
    """Global average pooling operation for spatial data.
        由于layers.GlobalAveragePooling2D返回张量是2维的  本类返回4维
    """

    def call(self, inputs):
        return backend.mean(inputs, axis=[1, 2], keepdims=True)
