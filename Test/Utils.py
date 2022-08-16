import os
import sys
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 训练集 & 测试集 数据读取
def image_reader(batch_size, im_height, im_width, image_path="./data_set/dateset_after_split"):
    """
    :param image_path: flower data set path
    :return:
    """
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    # 检查这两个路径是否存在
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # 图像归一化预处理
    def pre_function(img):
        img = img / 255.
        img = (img - 0.5) * 2.0
        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                               horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,  # 产生batch数据
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_train = train_data_gen.n
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))
    return train_data_gen, val_data_gen


# 测试脚本
def TestScript(model, batch_size=10, im_height=224, im_width=224, epochs=30, learning_rate=0.0005):
    """
    :param model:  # e.g. ResNeXt101(classes=5)
    :return: None
    """
    # 获得训练集和测试集
    train_data_gen, val_data_gen = image_reader(batch_size, im_height, im_width,
                                                image_path="./data_set/dateset_after_split")
    total_train, total_val = train_data_gen.n, val_data_gen.n  # 训练集和测试集的数量

    # 创建模型
    model.build(input_shape=(batch_size, 224, 224, 3))
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()  # 打印模型网络结构

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(inputs=images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        train_accuracy.update_state(labels, output)

    @tf.function
    def val_step(images, labels):
        output = model(inputs=images, training=False)
        loss = loss_object(labels, output)

        val_loss.update_state(loss)
        val_accuracy.update_state(labels, output)

    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(range(total_train // batch_size), file=sys.stdout)
        for step in train_bar:
            train_images, train_labels = next(train_data_gen)  # 每次获取一个batch
            train_step(train_images, train_labels)
            # tf.enable_eager_execution() 加上这句才出数 开启紧急执行 启用动态图机制 这一句必须放在文件的开头
            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}  ".format(epoch + 1,
                                                                                   epochs,
                                                                                   train_loss.result(),
                                                                                   train_accuracy.result())

        # validate
        val_bar = tqdm(range(total_val // batch_size), file=sys.stdout)
        for step in val_bar:
            val_images, val_labels = next(val_data_gen)
            val_step(val_images, val_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}  ".format(epoch + 1,
                                                                                 epochs,
                                                                                 val_loss.result(),
                                                                                 val_accuracy.result())
