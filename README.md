# TensorFlowFeaturesExample
Extracting features from a tensor flow model for transfer learning
*******

Classifies between two image categories, using a tensorflow graph model.

Dependancies
*******
TensorFlow
Scikit-Learn
Numpy

Usage
*******
usage: transfer_learning.py [-h] [--first_class FIRST_CLASS]
                            [--second_class SECOND_CLASS]
                            [--graph_file GRAPH_FILE]
                            [--layer_name LAYER_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --first_class FIRST_CLASS
                        Absolute path to the first category images locations
  --second_class SECOND_CLASS
                        Absolute path to the second category images locations
  --graph_file GRAPH_FILE
                        Absolute path the graph degenition protobuf file
  --layer_name LAYER_NAME
                        Name of the layer to extract features from

