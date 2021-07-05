import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


    # 缩小像素点的范围
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()
    # 准备模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 二维数组扁平化，转为一维数组
        keras.layers.Dense(128, activation='relu'),  # 128个神经元
        keras.layers.Dense(10)  # 十种东西，十种分类
    ])
    # 编译模型
    model.compile(optimizer='adam',  # 优化器
                  # 损失函数，最小化
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # 指标，用于监控训练和测试
                  metrics=['accuracy'])
    # 训练模型
    # 训练过程中会显示损失和准确率指标
    model.fit(train_images, train_labels, epochs=10)

    # 预测
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    predications = probability_model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predications[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predications[i], test_labels)
    plt.tight_layout()
    plt.show()

    # 评估准确率
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\n准确率：', test_acc, ' 损失率：', test_loss)