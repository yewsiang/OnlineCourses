import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    C, H, W = input_dim

    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale # (F, C, HH, WW)
    self.params['b1'] = np.zeros(num_filters) # (F, )

    self.params['W2'] = np.random.randn(num_filters, H / 2, W / 2, hidden_dim) * weight_scale # (F, H', W', Hidden)
    self.params['W2'] = self.params['W2'].reshape(-1, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim) # (Hidden, )

    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale # (Hidden, Classes)
    self.params['b3'] = np.zeros(num_classes) # (Classes, )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    conv1, conv1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine, affine_cache = affine_relu_forward(conv1, W2, b2)
    scores, scores_cache = affine_forward(affine, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))
    loss = data_loss + reg_loss

    daffine, dW3, db3 = affine_backward(dscores, scores_cache)
    dconv1, dW2, db2 = affine_relu_backward(daffine, affine_cache)
    dX, dW1, db1 = conv_relu_pool_backward(dconv1, conv1_cache)

    grads['W1'] = dW1 + self.reg * self.params['W1']
    grads['W2'] = dW2 + self.reg * self.params['W2']
    grads['W3'] = dW3 + self.reg * self.params['W3']

    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

# Added by YS for Last Part of Assignment 2
class FourLayerConvNet(object):
  """
  A four-layer convolutional network with the following architecture:
  
  (conv - batchnorm - relu - )x2 - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim

    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale # (F, C, HH, WW)
    self.params['b1'] = np.zeros(num_filters) # (F, )
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.randn(num_filters, num_filters, filter_size, filter_size) * weight_scale # (F, F, HH, WW)
    self.params['b2'] = np.zeros(num_filters) # (F, )
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)

    self.params['W3'] = np.random.randn(num_filters, H / 2, W / 2, hidden_dim) * weight_scale # (F, H', W', Hidden)
    self.params['W3'] = self.params['W3'].reshape(-1, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim) # (Hidden, )

    self.params['W4'] = np.random.randn(hidden_dim, num_classes) * weight_scale # (Hidden, Classes)
    self.params['b4'] = np.zeros(num_classes) # (Classes, )

    # 2 Batch norm layers
    self.bn_params = [{'mode': 'train'}, {'mode': 'train'}]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Batchnorm params
    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
      bn_param[mode] = mode

    scores = None


    conv1, conv1_cache = conv_batchnorm_relu_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param)    
    conv2, conv2_cache = conv_batchnorm_relu_forward(conv1, W2, b2, gamma2, beta2, conv_param, bn_param)
    pool1, pool1_cache = max_pool_forward_fast(conv2, pool_param)

    affine, affine_cache = affine_relu_forward(pool1, W3, b3)
    scores, scores_cache = affine_forward(affine, W4, b4)

    
    if y is None:
      return scores
    
    loss, grads = 0, {}


    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2) + np.sum(self.params['W4'] ** 2))
    loss = data_loss + reg_loss

    daffine, dW4, db4 = affine_backward(dscores, scores_cache)
    dpool1, dW3, db3 = affine_relu_backward(daffine, affine_cache)

    dconv2 = max_pool_backward_fast(dpool1, pool1_cache)
    dconv1, dW2, db2, dgamma2, dbeta2 = conv_batchnorm_relu_backward(dconv2, conv2_cache)
    dX, dW1, db1, dgamma1, dbeta1 = conv_batchnorm_relu_backward(dconv1, conv1_cache)

    grads['W1'] = dW1 + self.reg * self.params['W1']
    grads['W2'] = dW2 + self.reg * self.params['W2']
    grads['W3'] = dW3 + self.reg * self.params['W3']
    grads['W4'] = dW4 + self.reg * self.params['W4']

    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['b4'] = db4

    grads['gamma1'] = dgamma1
    grads['gamma2'] = dgamma2
    grads['beta1'] = dbeta1
    grads['beta2'] = dbeta2
    
    return loss, grads


pass
