#!/usr/bin/env python

import pdb
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

from utils.treebank import StanfordSentiment
import utils.glove as glove

from q3_sgd import load_saved_params, sgd

import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import confusion_matrix


def getArguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", dest="pretrained", action="store_true",
                       help="Use pretrained GloVe vectors.")
    group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
                       help="Use your vectors from q3.")
    return parser.parse_args()


def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE
    wordVectorsIdx = [tokens[word] for word in sentence]
    wordVectorsForSentence = [wordVectors[idx] for idx in wordVectorsIdx]
    wordVectorsForSentence = np.array(wordVectorsForSentence)
    sentVector = np.mean(wordVectorsForSentence, axis=0)
    ### END YOUR CODE

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


def chooseBestModel(results):
    """Choose the best model based on dev set performance.

    Arguments:
    results -- A list of python dictionaries of the following format:
        {
            "reg": regularization,
            "clf": classifier,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        }

    Each dictionary represents the performance of one model.

    Returns:
    Your chosen result dictionary.
    """
    bestResult = None

    ### YOUR CODE HERE
    bestDevAcc = -1
    for result in results:
        if result["dev"] > bestDevAcc:
            bestDevAcc = result["dev"]
            bestResult = result
    ### END YOUR CODE

    return bestResult


def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def plotParamVsAccuracy(params, results, filename):
    """ Make a plot of regularization vs accuracy """
    plt.plot(params, [x["train"] for x in results])
    plt.plot(params, [x["dev"] for x in results])
    plt.xscale('log')
    plt.xlabel("parameter")
    plt.ylabel("accuracy")
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(filename)


def outputConfusionMatrix(labels, pred, filename):
    """ Generate a confusion matrix """
    cm = confusion_matrix(labels, pred, labels=range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["- -", "-", "neut", "+", "+ +"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)


def outputPredictions(dataset, labels, pred, filename):
    """ Write the predictions to file """
    with open(filename, "w") as f:
        print >> f, "True\tPredicted\tText"
        for i in xrange(len(dataset)):
            print >> f, "%d\t%d\t%s" % (
                labels[i], pred[i], " ".join(dataset[i][0]))

def RNN(x, numHidden):
    # Define a lstm cell with tensorflow
    lstmCell = rnn.BasicLSTMCell(numHidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstmCell, x, dtype=tf.float32)

    return outputs

def BiRNN(x, numHidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(numHidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(numHidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
    return outputs

class SentenceLSTMModel(object):
    def __init__(self, numHidden, timeSteps, dimVectors, numClasses, rnnType="BiLSTM"):
        
        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([numHidden, numClasses]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([numClasses]))
        }

        # tf Graph input
        self.X_ph = tf.placeholder("float", [None, timeSteps, dimVectors])
        self.y_ph = tf.placeholder(tf.int32, [None])

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(self.X_ph, timeSteps, 1)

        # Recurrent network through the word vectors of a sentence
        if rnnType == "LSTM":
            outputs = RNN(x, numHidden)
        elif rnnType == "BiLSTM":
            outputs = BiRNN(x, numHidden / 2)
        else:
            raise NotImplementedError("No such RNN type")

        # Linear activation, using rnn inner loop last output
        self.logits = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        self.prediction = tf.nn.softmax(self.logits)

    def fit(self, X, y, trainingSteps, learningRate, batchSize=128, displayStep=200):

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.y_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        train_op = optimizer.minimize(loss_op)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Initialize session
        sess = tf.Session()
        self.sess = sess

        # Run the initializer
        sess.run(init)

        for step in range(1, trainingSteps+1):
            sampleId = np.random.choice(len(X), batchSize)
            X_batch, y_batch = X[sampleId,...], y[sampleId,...]

            # Run optimization op (backprop)
            feed_dict = {
                self.X_ph: X_batch, 
                self.y_ph: y_batch 
             }
            sess.run(train_op, feed_dict=feed_dict)
            if step % displayStep == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss = sess.run(loss_op, feed_dict=feed_dict)
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))

        print("Optimization Finished!")

    def predict(self, X, batchSize=128):
        pred = np.zeros((len(X)))
        numBatches = len(X) // batchSize + (len(X) % batchSize > 0)
        # pdb.set_trace()
        for i in range(numBatches):
            startId = i * batchSize
            endId = min((i + 1) * batchSize, len(X))
            # pdb.set_trace()
            X_batch = X[startId:endId,...]
            feed_dict = { 
                self.X_ph: X_batch 
            }
            pred_batch = self.sess.run(self.prediction, feed_dict=feed_dict)
            pred[startId:endId] = np.argmax(pred_batch, axis=1)
        return pred

def main(args):
    """ Train a model to do sentiment analyis"""

    # Load the dataset
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    if args.yourvectors:
        _, wordVectors, _ = load_saved_params()
        wordVectors = np.concatenate(
            (wordVectors[:nWords,:], wordVectors[nWords:,:]),
            axis=1)
    elif args.pretrained:
        wordVectors = glove.loadWordVectors(tokens)
    dimVectors = wordVectors.shape[1]

    # Load the train set
    trainset = dataset.getTrainSentences()
    devset = dataset.getDevSentences()
    testset = dataset.getTestSentences()
    nTrain = len(trainset)
    nDev = len(devset)
    nTest = len(testset)
    nTrainSentenceLen = max([len(words) for words, label in trainset])
    nDevSentenceLen = max([len(words) for words, label in devset])
    nTestSentenceLen = max([len(words) for words, label in testset])
    nSentenceLen = max(nTrainSentenceLen, nDevSentenceLen, nTestSentenceLen)
    
    trainFeatures = np.zeros((nTrain, nSentenceLen, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)
    for i in xrange(nTrain):
        words, trainLabels[i] = trainset[i]
        sentenceWordVecs = np.array([wordVectors[tokens[word]] for word in words])
        sentenceLen = len(sentenceWordVecs)
        trainFeatures[i,:sentenceLen] = sentenceWordVecs

    # Prepare dev set features
    devFeatures = np.zeros((nDev, nSentenceLen, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in xrange(nDev):
        words, devLabels[i] = devset[i]
        sentenceWordVecs = np.array([wordVectors[tokens[word]] for word in words])
        sentenceLen = len(sentenceWordVecs)
        devFeatures[i,:sentenceLen] = sentenceWordVecs

    # Prepare test set features
    testFeatures = np.zeros((nTest, nSentenceLen, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in xrange(nTest):
        words, testLabels[i] = testset[i]
        sentenceWordVecs = np.array([wordVectors[tokens[word]] for word in words])
        sentenceLen = len(sentenceWordVecs)
        testFeatures[i,:sentenceLen] = sentenceWordVecs

    # Hyperparameters
    learningRate = 0.001
    trainingSteps = 2000
    batchSize = 128
    displayStep = 200

    # Network Parameters
    timeSteps = nSentenceLen # time steps
    numHidden = 256 # hidden layer num of features
    numClasses = 5 # SST total classes (5 sentiments)

    # We will save our results from each run
    results = []
    hyperparams = [1]
    for param in hyperparams:
        print "Training for param=%f" % param
        # Note: add a very small number to regularization to please the library
        model = SentenceLSTMModel(numHidden, timeSteps, dimVectors, numClasses)
        model.fit(trainFeatures, trainLabels, trainingSteps, learningRate, 
                  batchSize, displayStep)

        # Test on train set
        trainPred = model.predict(trainFeatures)
        trainAccuracy = accuracy(trainLabels, trainPred)
        print "Train accuracy (%%): %f" % trainAccuracy

        # Test on dev set
        devPred = model.predict(devFeatures)
        devAccuracy = accuracy(devLabels, devPred)
        print "Dev accuracy (%%): %f" % devAccuracy

        # Test on test set
        # Note: always running on test is poor style. Typically, you should
        # do this only after validation.
        testPred = model.predict(testFeatures)
        testAccuracy = accuracy(testLabels, testPred)
        print "Test accuracy (%%): %f" % testAccuracy

        results.append({
            "param": param,
            "model": model,
            "train": trainAccuracy,
            "dev"  : devAccuracy,
            "test" : testAccuracy})

    # Print the accuracies
    print ""
    print "=== Recap ==="
    print "Reg\t\tTrain\tDev\tTest"
    for result in results:
        print "%.2E\t%.3f\t%.3f\t%.3f" % (
            result["param"],
            result["train"],
            result["dev"],
            result["test"])
    print ""

    bestResult = chooseBestModel(results)
    print "Best parameter value: %0.2E" % bestResult["param"]
    print "Test accuracy (%%): %f" % bestResult["test"]

    # do some error analysis
    if args.pretrained:
        plotParamVsAccuracy(hyperparams, results, "q4_lstm_param_v_acc.png")
        outputConfusionMatrix(devLabels, devPred,
                              "q4_lstm_dev_conf.png")
        outputPredictions(devset, devLabels, devPred,
                          "q4_lstm_dev_pred.txt")


if __name__ == "__main__":
    main(getArguments())
