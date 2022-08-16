import Utils
import sys
import tensorflow as tf
import os

from models.ResNeSt.ResNeSt import resnest50
from models.ShuffleNet.ShuffleNetV1 import ShuffleNetV1
from models.ShuffleNet.ShuffleNetV2 import ShuffleNetV2

rootPath = "/home/PyProject/algorithmlib"
sys.path.append(rootPath)
# from ResNeXt.ResNeXt import ResNeXt101
# from SE_ResNet.SE_ResNet import SE_ResNet50, SE_ResNet101, SE_ResNet152

# 使用GPU训练
# shell: CUDA_VISIBLE_DEVICES=0,  python  xxx.py
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

if __name__ == '__main__':
    # model = ResNeXt101(classes=5)
    # model = SE_ResNeXt50(classes=5)
    # Utils.TestScript(model, batch_size=20, im_height=224, im_width=224, epochs=30, learning_rate=0.0005)
    # model = SE_ResNeXt101(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0005)

    # model = SE_ResNet50(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
    # nohup python train.py > train_SE_ResNet50.log 2>&1 &

    # model = SE_ResNet101(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30)
    # # nohup python train.py > train_SE_ResNet101.log 2>&1 &
    #
    # model = SE_ResNet152(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
    # # nohup python train.py > train_SE_ResNet152.log 2>&1 &

    # model = SK_ResNet50(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
    # # nohup python train.py > train_SK_ResNet50.log 2>&1 &

    # model = SK_ResNet101(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
    # # nohup python train.py > train_SK_ResNet101.log 2>&1 &

    # model = SK_ResNet152(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
    # # nohup python train.py > train_SK_ResNet152.log 2>&1 &

    model = resnest50(classes=5)
    Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)

    # model = resnest101(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)

    # model = resnest152(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
    #
    # model = ShuffleNetV1(group=3, classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)

    # model = ShuffleNetV2(classes=5)
    # Utils.TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0001)
