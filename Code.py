import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Environment Initialization
grid_size = 10
red_position = (0, 4)  # Zero-indexed (1, 5)
blue_start_position = (9, 1)  # Zero-indexed (10, 2)
obstacles = [(4, 1), (4, 2), (4, 3), (6, 3), (2, 5), (2, 6), (2, 7)]
initial_power = 100
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, initial_power + 1, len(actions)))  # (grid_x, grid_y, power, actions)

# Define reward function
def get_reward(new_position, current_power):
    if new_position == red_position:
        return 100  # Large reward for reaching the goal
    elif new_position in obstacles:
        return -10  # Large penalty for hitting an obstacle
    else:
        return -1  # Small penalty for moving

# Update Q-table using Q-learning algorithm
def update_q_table(state, action, reward, new_state, alpha, gamma):
    max_future_q = np.max(Q[new_state])
    current_q = Q[state][action]
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    Q[state][action] = new_q

# Training function
def train(episodes, alpha, gamma, epsilon, epsilon_decay):
    for episode in range(episodes):
        state = (blue_start_position[0], blue_start_position[1], initial_power)  # Initial state

        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(Q[state])  # Exploit: choose the best action from Q-table
            else:
                action = np.random.randint(0, len(actions))  # Explore: choose a random action

            new_position = (
                state[0] + action_dict[actions[action]][0],
                state[1] + action_dict[actions[action]][1]
            )
            new_position = (max(0, min(grid_size - 1, new_position[0])),
                            max(0, min(grid_size - 1, new_position[1])))

            new_power = state[2] - 1  # Deduct power for each move

            if new_position in obstacles:
                new_power -= 10  # Additional penalty for hitting an obstacle

            new_state = (new_position[0], new_position[1], max(0, new_power))
            reward = get_reward(new_position, new_power)

            if new_position == red_position or new_power <= 0:
                done = True  # Episode ends when reaching the goal or running out of power

            update_q_table(state, action, reward, new_state, alpha, gamma)
            state = new_state

            if epsilon > 0.1:
                epsilon *= epsilon_decay  # Decay epsilon

    return Q

# Training parameters
alpha = 0.1           # Learning rate
gamma = 0.99          # Discount factor
epsilon = 1.0         # Exploration factor
epsilon_decay = 0.999 # Slower decay
episodes = 500000     # Increased number of episodes

# Train the Q-learning model
Q = train(episodes, alpha, gamma, epsilon, epsilon_decay)

# Get policy from Q-table
def get_policy(Q):
    policy = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)
    for x in range(grid_size):
        for y in range(grid_size):
            for power in range(initial_power + 1):
                policy[x, y, power] = np.argmax(Q[x, y, power])
    return policy

policy = get_policy(Q)

# Simulate game using the policy
def simulate_game(policy):
    state = (blue_start_position[0], blue_start_position[1], initial_power)
    path = [state]
    while state[:2] != red_position and state[2] > 0:
        action = policy[state]
        new_position = (
            state[0] + action_dict[actions[action]][0],
            state[1] + action_dict[actions[action]][1]
        )
        new_position = (max(0, min(grid_size - 1, new_position[0])),
                        max(0, min(grid_size - 1, new_position[1])))
        new_power = state[2] - 1
        if new_position in obstacles:
            new_power -= 10
        state = (new_position[0], new_position[1], max(0, new_power))
        path.append(state)
    return path

path = simulate_game(policy)

# Print the path
print("\nPath taken by the blue agent :\n")
for step in path:
    print(f"Position: ({step[0]+1}, {step[1]+1}), Power: {step[2]}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Visualization of Initial State
ax1.set_xlim(0, grid_size)
ax1.set_ylim(0, grid_size)
ax1.set_xticks(np.arange(1, grid_size + 1, 1))
ax1.set_yticks(np.arange(0, grid_size, 1))
ax1.grid()
ax1.set_title("Initial State\n\nPosition ≡  (row, col)", fontsize=12, fontweight='bold', pad=20)

# Draw the red agent
ax1.add_patch(patches.Rectangle((red_position[1], grid_size - red_position[0] - 1), 1, 1, fill=True, color='red'))

# Draw the blue agent's starting position
ax1.add_patch(patches.Rectangle((blue_start_position[1], grid_size - blue_start_position[0] - 1), 1, 1, fill=True, color='blue', alpha=0.3))

# Draw the obstacles
for obs in obstacles:
    ax1.add_patch(patches.Rectangle((obs[1], grid_size - obs[0] - 1), 1, 1, fill=True, color='black'))

# Reverse the y-axis to start from the bottom
ax1.invert_yaxis()

# Set x-ticks and y-ticks to match the grid and label them from 1 to 10
ax1.set_xticks(np.arange(1, grid_size + 1, 1))
ax1.set_xticklabels(np.arange(1, grid_size + 1, 1))  # X-axis labeled from 1 to 10
ax1.set_yticks(np.arange(0, grid_size, 1))
ax1.set_yticklabels(np.arange(grid_size, 0, -1))  # Y-axis labeled from 1 to 10, bottom to top

# Visualization of Most Optimal Path
ax2.set_xlim(0, grid_size)
ax2.set_ylim(0, grid_size)
ax2.set_xticks(np.arange(1, grid_size + 1, 1))
ax2.set_yticks(np.arange(0, grid_size, 1))
ax2.grid()
ax2.set_title("Final State", fontsize=12, fontweight='bold', pad=20)

# Draw the red agent
ax2.add_patch(patches.Rectangle((red_position[1], grid_size - red_position[0] - 1), 1, 1, fill=True, color='red'))

# Draw the blue agent's path
for step in path:
    ax2.add_patch(patches.Rectangle((step[1], grid_size - step[0] - 1), 1, 1, fill=True, color='blue', alpha=0.3))

# Draw the obstacles
for obs in obstacles:
    ax2.add_patch(patches.Rectangle((obs[1], grid_size - obs[0] - 1), 1, 1, fill=True, color='black'))

# Reverse the y-axis to start from the bottom
ax2.invert_yaxis()

# Set x-ticks and y-ticks to match the grid and label them from 1 to 10
ax2.set_xticks(np.arange(1, grid_size + 1, 1))
ax2.set_xticklabels(np.arange(1, grid_size + 1, 1))  # X-axis labeled from 1 to 10
ax2.set_yticks(np.arange(0, grid_size, 1))
ax2.set_yticklabels(np.arange(grid_size, 0, -1))  # Y-axis labeled from 1 to 10, bottom to top

plt.show()

