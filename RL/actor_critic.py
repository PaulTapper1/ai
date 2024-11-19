# code from https://gist.github.com/Ashioto/10ec680395db48ddac1ad848f5f7382c#file-actorcritic-py
# see continuous mountain car https://github.com/openai/gym/wiki/Leaderboard

#!pip install gymnasium
#!pip install tensorflow

import tensorflow as tf
import numpy as np
import os
import gymnasium as gym
import time
import sklearn
import itertools
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


env = gym.envs.make("MountainCarContinuous-v0") # https://github.com/openai/gym/wiki/MountainCarContinuous-v0
# video_dir = os.path.abspath("./videos")
# if not os.path.exists(video_dir):
    # os.makedirs(video_dir)
# env = gym.wrappers.RecordVideo(env, video_dir, force=True)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def process_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


class PolicyEstimator:
    def __init__(self, env, lamb=1e-5, learning_rate=0.01, scope="policy_estimator"):
        self.env = env
        self.lamb = lamb
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.compat.v1.placeholder(tf.float32, [400], name="state")

        # Use tf.keras.layers.Dense instead of tf.compat.v1.layers.dense
        self.mu_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal()
        )
        # Apply the layer to the input tensor
        self.mu = self.mu_layer(tf.expand_dims(self.state, 0))
        self.mu = tf.squeeze(self.mu)

         # Use tf.keras.layers.Dense instead of tf.compat.v1.layers.dense
        self.sigma_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal()
        )
        # Apply the layer to the input tensor
        self.sigma = self.sigma_layer(tf.expand_dims(self.state, 0))
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.norm_dist = tf.compat.v1.distributions.Normal(self.mu, self.sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])

    def _build_train_op(self):
        self.action_train = tf.compat.v1.placeholder(tf.float32, name="action_train")
        self.advantage_train = tf.compat.v1.placeholder(tf.float32, name="advantage_train")

        self.loss = -tf.compat.v1.log(
            self.norm_dist.prob(self.action_train) + 1e-5) * self.advantage_train - self.lamb * self.norm_dist.entropy()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        feed_dict = {self.state: process_state(state)}
        # Return the action as a 1-element list
        return [sess.run(self.action, feed_dict=feed_dict)] # Change here

    def update(self, state, action, advantage, sess):
        feed_dict = {
            self.state: process_state(state),
            self.action_train: action,
            self.advantage_train: advantage
        }
        sess.run([self.train_op], feed_dict=feed_dict)


class ValueEstimator:
    def __init__(self, env, learning_rate=0.01, scope="value_estimator"):
        self.env = env
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.compat.v1.placeholder(tf.float32, [400], name="state")

        # Use tf.keras.layers.Dense instead of tf.compat.v1.layers.dense
        self.value_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal()
        )
        # Apply the layer to the input tensor
        self.value = self.value_layer(tf.expand_dims(self.state, 0))
        self.value = tf.squeeze(self.value)

    def _build_train_op(self):
        self.target = tf.compat.v1.placeholder(tf.float32, name="target")
        # Replace tf.squared_difference with tf.math.squared_difference
        self.loss = tf.reduce_mean(tf.math.squared_difference(self.value, self.target)) 
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: process_state(state)})

    def update(self, state, target, sess):
        feed_dict = {
            self.state: process_state(state),
            self.target: target
        }
        sess.run([self.train_op], feed_dict=feed_dict)


@exec_time
def actor_critic(episodes=100, gamma=0.95, display=False, lamb=1e-5, policy_lr=0.001, value_lr=0.1):
    # Disable eager execution
    tf.compat.v1.disable_eager_execution() 
    # tf.reset_default_graph() # This line is removed as it's not needed in TF 2.x
    tf.compat.v1.reset_default_graph() # Use tf.compat.v1.reset_default_graph() for compatibility with TF 1.x code
    policy_estimator = PolicyEstimator(env, lamb=lamb, learning_rate=policy_lr)
    value_estimator = ValueEstimator(env, learning_rate=value_lr)
    
    with tf.compat.v1.Session() as sess: # Use tf.compat.v1.Session() for compatibility with TF 1.x code
        sess.run(tf.compat.v1.global_variables_initializer()) # Use tf.compat.v1.global_variables_initializer()
        stats = []
        for i_episode in range(episodes):
            state = env.reset()
            if isinstance(state, tuple): # check if env.reset() returns a tuple (state, info)
                state = state[0]  # extract the state from the tuple
            reward_total = 0
            for t in itertools.count():
                action = policy_estimator.predict(state, sess)
                #next_state, reward, done, _ = env.step(action[0]) # extract the action from the list
                # Change here: Use * to unpack any extra values returned by env.step()
                step_result = env.step(action[0]) # extract the action from the list
                next_state = step_result[0]
                reward = step_result[1]
                done = step_result[2]
                # Check if there are more than 3 values returned 
                # and assign any extra to the _ variable
                _ = step_result[3:]


                reward_total += reward

                if display:
                    env.render()

                target = reward + gamma * value_estimator.predict(next_state, sess)
                td_error = target - value_estimator.predict(state, sess)

                if i_episode < 20:
                    policy_estimator.update(state, action[0], advantage=td_error, sess=sess) # extract the action from the list
                    value_estimator.update(state, target, sess=sess)

                if done:
                    break
                state = next_state
            stats.append(reward_total)
            if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
                print(np.mean(stats[-100:]))
                print("Solved")
            print("Episode: {}, reward: {}.".format(i_episode, reward_total))
        return np.mean(stats[-100:])


if __name__ == "__main__":
    policy_lr, value_lr, lamb, gamma = [0.0001, 0.00046415888336127773, 2.782559402207126e-05, 0.98999999999999999]
    loss = actor_critic(episodes=1000, gamma=gamma, display=False, lamb=lamb, policy_lr=policy_lr, value_lr=value_lr)
    print(-loss)
    env.close()
