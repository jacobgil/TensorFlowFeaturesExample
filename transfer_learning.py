import os.path
import sys
import glob
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn import svm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('first_class', '', """Absolute path to the first category images locations""")
tf.app.flags.DEFINE_string('second_class', '', """Absolute path to the second category images locations""")
tf.app.flags.DEFINE_string('graph_file', '', """Absolute path the graph degenition protobuf file""")
tf.app.flags.DEFINE_string('layer_name', 'pool_3:0', """Name of the layer to extract features from""")

def create_graph(graph_file):
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def extract_features(image, tensor_name):
  image_data = gfile.FastGFile(image, 'rb').read()
  print "Extracting features for", image
  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name(tensor_name)
    features = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    features = np.squeeze(features)
    return features

def shuffle_data(features, labels):
  new_features, new_labels = [], []
  index_shuf = range(len(features))
  shuffle(index_shuf)
  for i in index_shuf:
      new_features.append(features[i])
      new_labels.append(labels[i])

  return new_features, new_labels

def get_jpeg_dataset(A_DIR, B_DIR, tensor_name):
  CLASS_A_FEATURES = [extract_features(f, tensor_name) 
                      for f in glob.glob(A_DIR + "/*.jpg")]
  CLASS_B_FEATURES = [extract_features(f, tensor_name)
                      for f in glob.glob(B_DIR + "/*.jpg")]

  features = CLASS_A_FEATURES + CLASS_B_FEATURES
  labels = [0] * len(CLASS_A_FEATURES) + [1] * len(CLASS_B_FEATURES)
  
  return shuffle_data(features, labels)

def main(_):
  create_graph(FLAGS.graph_file)
  
  x, y = get_jpeg_dataset(FLAGS.first_class, FLAGS.second_class, FLAGS.layer_name)
  TRAINING_PORTION = 0.4
  l = int(len(y) * TRAINING_PORTION)

  x_train, y_train = x[: l], y[: l]
  x_test, y_test = x[l : ], y[l : ]

  clf = svm.SVC()
  clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)
  print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)  

if __name__ == '__main__':
  tf.app.run()
