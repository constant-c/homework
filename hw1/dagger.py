import os
from random import random

import gym
# os.chdir(os.path.join(os.path.abspath(os.path.curdir), "hw1/"))
import load_policy
import numpy as np
import tensorflow as tf
import tf_util
from keras import Sequential
from keras.layers import Dense


def roll_out(env_name, policy_fn, render=False, max_timesteps=1000):
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    observations = []
    actions = []

    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        action = policy_fn(obs[None, :])
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
        if render:
            env.render()
        if steps >= max_steps:
            break

    print('return: ', totalr)

    return {'observations': np.array(observations),
            'actions': np.array(actions),
            'returns': np.array([totalr])}


def build_network(output_units, hidden_layers, units_per_layer):
    model = Sequential()
    for _ in range(hidden_layers):
        model.add(Dense(units=units_per_layer, activation='relu'))

    model.add(Dense(units=output_units))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def report_results(expert_returns, novice_returns, config):
    with open('dagger_results.txt', 'a') as f:
        f.write('{}: {} demonstrations, network size {}, {} epochs \n'.format(config['env'], config['demos'],
                                                                              config['nn_size'], config['epochs']))
        f.write('-----------------------------------\n')
        f.write('Agent | Mean Return | Std of Return\n')
        f.write('===================================\n')
        f.write('Expert | {:.2f} | {:.2f}\n'.format(np.mean(expert_returns), np.std(expert_returns)))
        f.write('Novice | {:.2f} | {:.2f}\n'.format(np.mean(novice_returns), np.std(novice_returns)))
        f.write('-----------------------------------\n\n')


def dagger(env_name, num_rollouts=20, max_timesteps=1000, beta=1, nn_hidden_layers=2, nn_units_per_layer=64, epochs=5):
    data_set = None

    print('loading and building expert policy')
    expert_policy = load_policy.load_policy('experts/{}-v1.pkl'.format(env_name.split('-')[0]))
    print('loaded and built')

    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    novice_policy = build_network(action_dim, nn_hidden_layers, nn_units_per_layer)

    def get_beta():
        return pow(0.5, i)

    policy = lambda x: expert_policy(x) if get_beta() >= random() else novice_policy.predict(x)
    for i in range(num_rollouts):
        print("Rollout: %i/%i" % (i, num_rollouts))

        with tf.Session():
            tf_util.initialize()
            data = roll_out(env_name, policy, max_timesteps=max_timesteps)
            data['actions'] = expert_policy(data['observations'])

        if data_set is None:
            data_set = data
        else:
            for key in ('observations', 'actions', 'returns'):
                data_set[key] = np.vstack((data_set[key], data[key]))

        novice_policy.fit(data_set['observations'], data_set['actions'], epochs=epochs, batch_size=64)

    config = {'env': env_name,
              'demos': data_set['observations'].shape[0],
              'nn_size': '({}x{})'.format(nn_hidden_layers, nn_units_per_layer),
              'epochs': epochs}
    report_results(data_set['returns'][0], data_set['returns'], config)


if __name__ == "__main__":
    if os.path.exists('dagger_results.txt'):
        os.remove('dagger_results.txt')

    dagger('Ant-v2', num_rollouts=2, )

    # for env in ('Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2'):
    #     main(env, expert_rollouts=200)

    # # Grid search over neural net size to see if there is a better config for Humanoid
    # for num_rollouts in (200, 500):
    #     for num_layers in (1, 2, 3):
    #         for layer_size in (32, 64, 128, 256, 512, 1024):
    #             main('Humanoid-v2', num_rollouts, num_layers, layer_size)
