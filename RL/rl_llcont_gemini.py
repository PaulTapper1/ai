import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation for continuous actions

# Define the experience replay memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return torch.tensor(state, dtype=torch.float), torch.tensor(action, dtype=torch.float), torch.tensor(reward, dtype=torch.float), torch.tensor(next_state, dtype=torch.float), torch.tensor(done, dtype=torch.float)

    def __len__(self):
        return len(self.buffer)

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(capacity=100000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.tau = 0.005  # Target network update rate

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action within action space bounds
            return np.random.uniform(-1, 1, size=self.action_size)
        else:
            # Exploit: choose the best action according to the policy network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                return self.policy_net(state_tensor).numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # Get predicted Q-values for the current state-action pairs
        q_values = self.policy_net(state)
        q_value = torch.bmm(q_values.unsqueeze(1), action.unsqueeze(2)).squeeze()  
        
        # Get target Q-values for the next state
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            target_q_value = reward + (1 - done) * self.gamma * next_q_values.max(1)[0] 

        # Compute the loss
        loss = F.mse_loss(q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.policy_net, self.target_net, self.tau)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Create the environment
env = gym.make('LunarLanderContinuous-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Create the DQN agent
agent = DQNAgent(state_size, action_size)

# Training loop
num_episodes = 1000 
for episode in range(num_episodes):
    state = env.reset()[0]  # Reset environment
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)  # Get action from agent
        next_state, reward, done, _, _ = env.step(action) # Take action in environment
        agent.remember(state, action, reward, next_state, done) # Store experience in memory
        agent.learn() # Update agent's policy
        state = next_state
        total_reward += reward

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Save the trained model (optional)
# torch.save(agent.policy_net.state_dict(), 'lunar_lander_continuous_dqn.pth')

env.close()