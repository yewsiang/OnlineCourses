import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Shapes
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    X_i = X[i]
    scores = np.dot(X_i, W)

    # Modify the values to prevent numerical instability due to large values
    normalization_term = np.max(scores)
    scores -= normalization_term

    scores = np.exp(scores)
    correct_class_score = scores[y[i]]

    # Loss: L = -log (e^(s_yi) / sum_j (e^(s_j)))
    loss += -np.log(correct_class_score / np.sum(scores))

    # Gradient
    # (Derivation)
    # L = -XW_yi + log(sum_j(e^{XW_j}))
    #
    # (Formula)
    # d(log_e( f(x)) )/dx = f'(x) / f(x)
    #
    # dL/dW_yi = -X + Xe^{XW_yi} / sum_j(e^{XW_j}) 
    #          =   X(  e^{XW_yi} / sum_j(e^{XW_j}) - 1)
    # dL/dW_j  =      Xe^{XW_j}  / sum_j(e^{XW_j}) 
    #          =   X(  e^{XW_j}  / sum_j(e^{XW_j}) - 0)
    # Therefore,
    # dL/dW_k  =   X(  e^{XW_k}  / sum_j(e^{XW_j}) - (k == correct_class))
    #
    for j in xrange(num_classes):
      dW[:,j] += X_i * (scores[j] / np.sum(scores) - (j == y[i]))

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  # Average
  loss /= num_train
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Shapes
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = np.dot(X, W)

  # Modify the values to prevent numerical instability due to large values
  normalization_term = np.max(scores, axis=1)
  normalization_term = np.tile(normalization_term, (num_classes, 1)).T
  scores -= normalization_term
  scores = np.exp(scores)
  correct_class_scores = scores[range(0, num_train), y]

  # Loss: L = -log (e^(s_yi) / sum_j (e^(s_j)))
  loss = np.sum(-np.log(correct_class_scores / np.sum(scores, axis=1)))

  # Gradient
  # For each class j: dW[:,j] += X_i * (scores[j] / np.sum(scores) - (j == y[i]))
  # Segment: (scores[j] / np.sum(scores)) 
  softmax = np.divide(scores, np.tile(np.sum(scores, axis=1), (num_classes, 1)).T)

  # Segment: (j == y[i])
  correct_class = np.zeros(softmax.shape)
  correct_class[range(0, correct_class.shape[0]),y] = 1

  # Segment: (scores[j] / np.sum(scores) - (j == y[i]))
  intermediate = softmax - correct_class

  # Final
  dW = np.dot(X.T, intermediate)

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  # Average
  loss /= num_train
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

