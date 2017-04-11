import tensorflow as tf
import numpy as np
import pandas as pd
import os.path as osp

from tensorflow import flags


FLAGS = flags.FLAGS


if __name__ == "__main__":
  flags.DEFINE_string("data_dir", "data/",
      "Directory for storing input data.")
  flags.DEFINE_string("result_dir", "results/",
      "Directory for storing results.")
  flags.DEFINE_integer("batch_size", 100,
      "How many examples to process per batch for training.")


def dense_to_one_hot(labels, num_classes):
  num_labels = labels.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels] = 1
  return labels_one_hot


def main(unused_argv):
  train_data = pd.read_csv(osp.join(FLAGS.data_dir, "train.csv")).values
  train_images = train_data[:,1:]
  train_images = train_images.astype(np.float)
  train_images = np.multiply(train_images, 1.0 / 255.0)
  labels = np.squeeze(train_data[:,:1])
  num_classes = np.unique(labels).shape[0]
  labels = dense_to_one_hot(labels, num_classes)

  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  predict = tf.argmax(y, 1)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  epochs_completed = 0
  index_in_epoch = 0
  for _ in range(1000):
    start = index_in_epoch
    index_in_epoch += FLAGS.batch_size
    if index_in_epoch > train_images.shape[0]:
      epochs_completed += 1
      perm = np.arange(train_images.shape[0])
      np.random.shuffle(perm)
      train_images = train_images[perm]
      labels = labels[perm]
      start = 0
      index_in_epoch = FLAGS.batch_size
      assert FLAGS.batch_size <= train_images.shape[0]
    end = index_in_epoch
    batch_xs = train_images[start:end]
    batch_ys = labels[start:end]
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

  test_images = pd.read_csv(osp.join(FLAGS.data_dir, "test.csv")).values
  test_images = test_images.astype(np.float)
  test_images = np.multiply(test_images, 1.0 / 255.0)
  predicted_labels = np.zeros(test_images.shape[0])
  for i in range(test_images.shape[0] // FLAGS.batch_size):
    predicted_labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size] = sess.run(predict, feed_dict = {x: test_images[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]})
  predicted_labels = predicted_labels.astype(np.int)
  sumbmission = pd.DataFrame(data={"ImageId": (np.arange(test_images.shape[0]) + 1), "Label": predicted_labels})
  sumbmission.to_csv(osp.join(FLAGS.result_dir, "softmax.csv"), index = False)


if __name__ == "__main__":
  tf.app.run()

