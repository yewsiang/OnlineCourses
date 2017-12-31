#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import gym
import time
import pickle
import numpy as np
import tensorflow as tf

import tf_util
import load_policy
from run_bc import get_placeholders, get_model, get_loss

slim = tf.contrib.slim

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--save_dest_dir', type=str, default='bc_ckpt')
    parser.add_argument('--pause', type=float, default=0.05)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of model')
    args = parser.parse_args()

    if not os.path.exists(args.save_dest_dir): os.mkdir(args.save_dest_dir)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

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
                action = policy_fn(obs[None,:])
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

        # ========================= SECTION 2 =========================
        # Train Behavioural Cloning Model
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

                loss_sum = 0
                for step in range(100000):

                    sample = np.random.randint(0,N,[BATCH_SIZE])
                    X = expert_data['observations'][sample,:]
                    y = expert_data['actions'][sample,:,:].reshape(BATCH_SIZE,-1)

                    feed_dict = { obs: X, actions: y, is_training: True }
                    l, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    loss_sum += l

                    if step % 1000 == 0:
                        print("[Step %dK] Loss: %.4f" % (step / 1000, loss_sum / 1000))
                        saver.save(sess, os.path.join(args.save_dest_dir, args.envname))
                        loss_sum = 0

if __name__ == '__main__':
    main()
