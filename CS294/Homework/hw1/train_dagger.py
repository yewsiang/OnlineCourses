#!/usr/bin/env python

"""
Run the expert policy first to generate roll-out data for behavioral cloning training.
Then, run this script to train using DAgger on the expert data + expert policy.
Example usage:
    python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20
    python train_dagger.py experts/Hopper-v1.pkl experts/Hopper-v1-data.pkl Hopper-v1
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

def minibatch(expert_data, dagger_data, batch_size):
    """
    Sample batch_size data from expert_data and dagger_data according to some
    sampling scheme.
    """
    N_exp = len(expert_data['observations'])
    N_dag = len(dagger_data['observations'])

    if N_dag == 0:
        sample = np.random.randint(0, N_exp, [batch_size])
        X = expert_data['observations'][sample,:]
        y = expert_data['actions'][sample,:,:].reshape(batch_size,-1)
    else:
        # Sample equally
        assert(batch_size % 2 == 0)
        B = int(batch_size / 2)
        sample_exp = np.random.randint(0, N_exp, [B])
        X_exp = expert_data['observations'][sample_exp,:]
        y_exp = expert_data['actions'][sample_exp,:,:].reshape(B,-1)

        sample_dag = np.random.randint(0, N_dag, [B])
        X_dag = dagger_data['observations'][sample_dag,:]
        y_dag = dagger_data['actions'][sample_dag,:,:].reshape(B,-1) 

        X = np.concatenate([X_exp, X_dag], axis=0)
        y = np.concatenate([y_exp, y_dag], axis=0)       
    return X, y

def rollout(env, our_policy, expert_policy, num_rollouts, max_steps):
    """
    Rollout using our policy and use the expert policy to get action labels for
    the encountered observations.
    """
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            our_action = our_policy(obs[None,:])
            expert_action = expert_policy(obs[None,:])
            observations.append(obs)
            # Keep the expert action but execute our own action
            actions.append(expert_action)
            obs, r, done, _ = env.step(our_action)

            totalr += r
            steps += 1
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    return returns, observations, actions

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--save_dest_dir', type=str, default='dag_ckpt')
    parser.add_argument('--pause', type=float, default=0.05)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of model')
    args = parser.parse_args()

    if not os.path.exists(args.save_dest_dir): 
        os.mkdir(args.save_dest_dir)

    # Environment for rollouts
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    # Loading expert data
    with open(args.expert_data_file, 'rb') as f:
        returns, expert_data = pickle.load(f)
    dagger_data = {
        'observations': [],
        'actions': []
    }

    # Train DAgger model
    BATCH_SIZE = 100
    N, input_dims  = expert_data['observations'].shape
    _, _, output_dims = expert_data['actions'].shape

    with tf.Graph().as_default():

        # Loading expert policy
        expert_policy = load_policy.load_policy(args.expert_policy_file)

        is_training = tf.placeholder(tf.bool, shape=())
        learning_rate = tf.Variable(args.learning_rate)

        # Initialize
        obs, actions = get_placeholders(BATCH_SIZE, input_dims, output_dims)
        pred_op = get_model(obs, output_dims, is_training)
        loss_op = get_loss(actions, pred_op)

        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss_op, opt)

        # Save checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Initialize
            sess.run(tf.global_variables_initializer())

            print(" ***** Training DAgger Agent *****")
            print(" Observations: %s" % str(expert_data['observations'].shape))
            print(" Actions     : %s" % str(expert_data['actions'].shape))

            losses = []
            loss_sum = 0
            num_steps = 30
            num_update_steps = 1000
            for step in range(num_steps):
                N_expert = len(expert_data['observations'])
                N_dagger = len(dagger_data['observations'])
                N_total = N_expert + N_dagger
                print("[Step %d]" % (step))
                print("Expert/Dagger data: %d/%d (%.1f%%/%.1f%%)" % (N_expert, N_dagger, N_expert*100./N_total, N_dagger*100./N_total))

                # Train on expert_data + dagger_data
                for _ in range(num_update_steps):
                    X, y = minibatch(expert_data, dagger_data, BATCH_SIZE)
                    feed_dict = { obs: X, actions: y, is_training: True }
                    l, _ = sess.run([loss_op, train_op], feed_dict=feed_dict)
                    loss_sum += l

                # Report loss
                print("Loss: %.4f" % (loss_sum / num_update_steps))
                saver.save(sess, os.path.join(args.save_dest_dir, args.envname))
                losses.append(loss_sum / num_update_steps)
                loss_sum = 0
                
                # Rollout our policy and use expert_policy to get labels
                def our_policy(new_obs):
                    # new_obs is of len (1,D) but obs is of (B,D)
                    # We duplicate to use the same placeholder
                    new_obs_exp = np.tile(new_obs, [BATCH_SIZE,1])
                    feed_dict = { obs: new_obs_exp, is_training: False }
                    action_exp = sess.run(pred_op, feed_dict=feed_dict)
                    # Keep only one of the actions
                    action = action_exp[:1,:]
                    return action

                new_returns, new_obs, new_actions = rollout(env, our_policy, 
                    expert_policy, args.num_rollouts, max_steps)

                # Aggregate data
                if len(dagger_data['observations']) == 0:
                    dagger_data['observations'] = np.array(new_obs)
                    dagger_data['actions'] = np.array(new_actions)
                else:
                    dagger_data['observations'] = np.concatenate([dagger_data['observations'], new_obs], axis=0)  
                    dagger_data['actions'] = np.concatenate([dagger_data['actions'], new_actions], axis=0)  

            plt.plot(range(1, num_steps + 1), losses)
            plt.savefig(os.path.join(args.save_dest_dir, args.envname) + '.png')

if __name__ == '__main__':
    main()
