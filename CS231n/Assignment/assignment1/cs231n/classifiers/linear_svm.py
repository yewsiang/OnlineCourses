import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] += -X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X, W)
  
  # Choose the scores of the correct class using the label "y"
  correct_class_score = np.choose(y, scores.T)
  correct_class_score = np.tile(correct_class_score, (scores.shape[1], 1))

  # Compute the margin matrix using the formula
  margin = scores - correct_class_score.T + 1
  margin = np.maximum(0, margin)
  # Set all correct class margin values to 0
  margin[range(0, num_train), y] = 0
  loss = np.sum(margin)

  # Get average
  loss /= num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # (N x C) array showing which margin is above 0
  margin_above_0 = (margin > 0) 
  # (N x 1) array showing number of times each training eg should be added
  num_above_0 = np.sum(margin_above_0, axis=1) 

  # (N x C) array where cth column = 1 if the label is class c. Else, it will be 0
  y2 = np.zeros((num_train, num_classes))
  y2[range(0, num_train), y] = 1
  # Multiply y2 row-wise by num_above_0 to add training egs multiple times
  y2 = np.multiply(y2, num_above_0[:, np.newaxis])
  # Multiply X.T with y2 to get dW from correct class
  # (D x C) array
  dW_due_to_correct_class = -1 * np.dot(X.T, y2)

  # (D x C) array
  dW_from_other_classes = np.dot(X.T, margin_above_0)

  # Get average
  dW = dW_due_to_correct_class + dW_from_other_classes
  dW /= num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
