#!/usr/bin/env python

"""
Run the expert policy first to generate roll-out data for behavioral cloning training.
Then, run this script to train on the expert data.
Example usage:
    python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20
    python train_bc.py experts/Hopper-v1-data.pkl Hopper-v1
"""

import pdb
import os
import gym
import time
import pickle
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tf_util
import load_policy
from run_bc import get_placeholders, get_model, get_loss

slim = tf.contrib.slim

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--save_dest_dir', type=str, default='bc_ckpt')
    parser.add_argument('--pause', type=float, default=0.05)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of model')
    args = parser.parse_args()

    if not os.path.exists(args.save_dest_dir): 
        os.mkdir(args.save_dest_dir)

    # Loading expert data
    with open(args.expert_data_file, 'rb') as f:
        returns, expert_data = pickle.load(f)

    # Train Behavioural Cloning model
    BATCH_SIZE = 100
    N, input_dims  = expert_data['observations'].shape
    _, _, output_dims = expert_data['actions'].shape

    with tf.Graph().as_default():

        is_training = tf.placeholder(tf.bool, shape=())
        learning_rate = tf.Variable(args.learning_rate)

        # Initialize
        obs, actions = get_placeholders(BATCH_SIZE, input_dims, output_dims)
        pred = get_model(obs, output_dims, is_training)
        loss = get_loss(actions, pred)

        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss, opt)

        # Save checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Initialize
            sess.run(tf.global_variables_initializer())

            print(" ***** Training Behavioural Cloning Agent *****")
            print(" Observations: %s" % str(expert_data['observations'].shape))
            print(" Actions     : %s" % str(expert_data['actions'].shape))

            losses = []
            loss_sum = 0
            num_steps = 30000
            for step in range(num_steps):

                sample = np.random.randint(0,N,[BATCH_SIZE])
                X = expert_data['observations'][sample,:]
                y = expert_data['actions'][sample,:,:].reshape(BATCH_SIZE,-1)

                feed_dict = { obs: X, actions: y, is_training: True }
                l, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                loss_sum += l

                if step % 1000 == 0 and step > 0:
                    print("[Step %dK] Loss: %.4f" % (step / 1000, loss_sum / 1000))
                    saver.save(sess, os.path.join(args.save_dest_dir, args.envname))
                    losses.append(loss_sum / 1000)
                    loss_sum = 0

            plt.plot(range(1, int(num_steps/1000)), losses)
            plt.savefig(os.path.join(args.save_dest_dir, args.envname) + '.png')

if __name__ == '__main__':
    main()
