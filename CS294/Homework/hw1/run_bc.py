#!/usr/bin/env python

import os
import gym
import time
import pickle
import numpy as np
import tensorflow as tf

import tf_util
import load_policy

slim = tf.contrib.slim

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--ckpt_dir', type=str, default='bc_ckpt')
    parser.add_argument('--pause', type=float, default=0.05)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of model')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        raise Exception('There is no pre-trained BC model.')

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    observation = env.reset()
    action = env.action_space.sample()

    input_dims  = observation.shape[0]
    output_dims = action.shape[0]

    with tf.Graph().as_default():

        is_training = tf.placeholder(tf.bool, shape=())
        learning_rate = tf.Variable(args.learning_rate)

        # Initialize BC Model
        obs_pl, action_pl = get_placeholders(1, input_dims, output_dims)
        pred = get_model(obs_pl, output_dims, is_training)
        loss = get_loss(action_pl, pred)

        # Initialize Expert Policy
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')

        with tf.Session() as sess:

            # For Expert Policy
            tf_util.initialize()

            # For BC Model
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(args.ckpt_dir, args.envname))

            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    if args.pause > 0:
                        time.sleep(args.pause)

                    # Use expert policy or BC model to choose actions
                    #action = policy_fn(obs[None,:])
                    feed_dict = { obs_pl : obs[None,:], is_training : False }
                    action = sess.run(pred, feed_dict=feed_dict)

                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)

                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}
        

# =============================== SECTION 2 ===============================

def get_placeholders(batch_size, input_dims, output_dims):
    observations = tf.placeholder(tf.float32, shape=(batch_size, input_dims))
    actions = tf.placeholder(tf.float32, shape=(batch_size, output_dims))
    return observations, actions

def get_model(obs, output_dims, is_training):
    """
    Function that takes in an observation, and returns the action to take.
    """
    with tf.variable_scope('Behavioural_Cloning'):
        with slim.arg_scope([slim.fully_connected], 
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            net = slim.fully_connected(obs, 512, scope='fc1')
            net = slim.fully_connected(net, 256, scope='fc2')
            net = slim.fully_connected(net, 128, scope='fc3')
            net = slim.fully_connected(net, 128, scope='fc4')
            net = slim.dropout(net, keep_prob=0.7, scope='dp4', is_training=is_training)
            net = slim.fully_connected(net, 128, scope='fc5')
            net = slim.dropout(net, keep_prob=0.7, scope='dp5', is_training=is_training)
        net = slim.fully_connected(net, output_dims, scope='fc6')
    return net

def get_loss(label, pred):
    """
    Calculates loss by difference between supposed action and prediction.
    """
    mse = tf.losses.huber_loss(label, pred)
    return mse


if __name__ == '__main__':
    main()
