
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN definition
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Parameters
input_dim = 4  # [object_position, object_distance, left_space, right_space]
output_dim = 3  # [Move Left, Move Right, No Move]
policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

memory = deque(maxlen=10000)
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1000

# Simulate environment

def generate_state():
    position = random.uniform(-1, 1)
    distance = random.uniform(0, 1)
    left_space = random.choice([0, 1])
    right_space = random.choice([0, 1])
    return [position, distance, left_space, right_space]

def compute_action(state):
    position, distance, left_space, right_space = state
    if distance < 0.2 or (left_space == 0 and right_space == 0):
        return 2  # No Move
    elif position < -0.1 and right_space:
        return 1  # Move Right
    elif position > 0.1 and left_space:
        return 0  # Move Left
    else:
        return 2  # No Move

def train_model():
    global EPSILON
    for episode in range(EPISODES):
        state = generate_state()
        action = compute_action(state)

        reward = 1 if action == compute_action(state) else -1
        next_state = generate_state()
        memory.append((state, action, reward, next_state))

        if len(memory) >= BATCH_SIZE:
            transitions = random.sample(memory, BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

            batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
            batch_action = torch.tensor(batch_action, dtype=torch.int64).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(device)

            q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze()
            next_q_values = target_net(batch_next_state).max(1)[0].detach()
            expected_q_values = batch_reward + GAMMA * next_q_values

            loss = criterion(q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {EPSILON:.3f}, Loss: {loss.item():.4f}")
        else:
            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {EPSILON:.3f}, Not enough memory to train.")

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

    torch.save(policy_net.state_dict(), "dodge_model.pth")
    print("\n Model training complete and saved as dodge_model.pth")

if __name__ == "__main__":
    train_model()