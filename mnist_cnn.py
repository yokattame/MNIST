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
  flags.DEFINE_integer("batch_size", 50,
      "How many examples to process per batch for training.")


def dense_to_one_hot(labels, num_classes):
  num_labels = labels.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels] = 1
  return labels_one_hot


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def main(unused_argv):
  train_data = pd.read_csv(osp.join(FLAGS.data_dir, "train.csv")).values
  train_images = train_data[:,1:]
  train_images = train_images.astype(np.float)
  train_images = np.multiply(train_images, 1.0 / 255.0)
  labels = np.squeeze(train_data[:,:1])
  num_classes = np.unique(labels).shape[0]
  labels = dense_to_one_hot(labels, num_classes)
  
  x = tf.placeholder(tf.float32, [None, 784])
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  #W_xx = tf.Variable(tf.zeros([784, 10]))
  #W_xx = weight_variable([784, 10])
  #b_xx = tf.Variable(tf.zeros([10]))
  #b_xx = bias_variable([10])
  #y = tf.matmul(x, W_xx) + b_xx
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  predict = tf.argmax(y, 1)
  correct_prediction = tf.equal(predict, tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  epochs_completed = 0
  index_in_epoch = 0
  for i in range(20000):
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
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

  test_images = pd.read_csv(osp.join(FLAGS.data_dir, "test.csv")).values
  test_images = test_images.astype(np.float)
  test_images = np.multiply(test_images, 1.0 / 255.0)
  predicted_labels = np.zeros(test_images.shape[0])
  for i in range(test_images.shape[0] // FLAGS.batch_size):
    predicted_labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size] = sess.run(predict, feed_dict={x: test_images[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size], keep_prob: 1.0})
  predicted_labels = predicted_labels.astype(np.int)
  sumbmission = pd.DataFrame(data={"ImageId": (np.arange(test_images.shape[0]) + 1), "Label": predicted_labels})
  sumbmission.to_csv(osp.join(FLAGS.result_dir, "cnn.csv"), index = False)


if __name__ == "__main__":
  tf.app.run()

