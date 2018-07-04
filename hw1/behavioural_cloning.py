import os
import pickle

import gym
# os.chdir(os.path.join(os.path.abspath(os.path.curdir), "hw1/"))
import load_policy
import numpy as np
import tensorflow as tf
import tf_util
from keras import Sequential
from keras.layers import Dense


def generate_expert_data(expert_policy_file, env_name, max_timesteps=None, num_rollouts=20):
    # See if we can load data from file
    data_pkl_path = expert_policy_file.replace(".pkl", "") + env_name + str(num_rollouts) + ".pkl"
    if os.path.exists(data_pkl_path):
        print('loading expert rollouts from file ' + data_pkl_path)
        with open(data_pkl_path, 'rb') as f:
            return pickle.load(f)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    # Roll out policy
    with tf.Session():
        tf_util.initialize()
        data = roll_out(env_name, policy_fn, max_timesteps=max_timesteps, num_rollouts=num_rollouts)

    # Save data for subsequent runs
    with open(data_pkl_path, 'wb') as f:
        print('saving data to file ' + data_pkl_path)
        pickle.dump(data, f)

    return data


def roll_out(env_name, policy_fn, render=False, max_timesteps=1000, num_rollouts=20):
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
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
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return {'observations': np.array(observations),
            'actions': np.array(actions),
            'returns': np.array(returns)}


def build_network():
    model = Sequential()
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=17))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def main(env_name):
    training_data = generate_expert_data('experts/Humanoid-v1.pkl', 'Humanoid-v2', max_timesteps=1500, num_rollouts=60)
    behavioural_clone = build_network()
    print('train clone')
    behavioural_clone.fit(training_data['observations'], training_data['actions'].reshape([-1, 17]), epochs=5,
                          batch_size=64)
    roll_out(env_name, policy_fn=lambda x: behavioural_clone.predict(x), render=True)


if __name__ == "__main__":
    main('Humanoid-v2')
