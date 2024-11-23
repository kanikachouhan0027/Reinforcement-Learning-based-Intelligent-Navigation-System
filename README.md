Design and implement a Intelligent-Navigation-System with help of a game using Q-learning and reinforcement learning where a blue agent must navigate a 10x10 grid to reach a static red agent in the most optimal and efficient way while avoiding obstacles and managing power consumption efficiently.(it can be thought of as the agent as tanks and power as fuel and the path as ground)

Game Description

Grid: The game is set on a 10x10 matrix representing the game board.

Agents:

Red Agent: Static, located at position (0, 4) on the grid (zero-indexed).

Blue Agent: Movable, starting at position (9, 1) on the grid (zero-indexed).

Obstacles: Fixed positions on the grid which the blue agent must avoid:
(4, 1), (4, 2), (4, 3)
(6, 3)
(2, 5), (2, 6), (2, 7)

Power: The blue agent starts with 100 power units. Moving to an adjacent grid cell costs 1 power unit, and hitting an obstacle costs 10 power units.

Objective: The blue agent needs to navigate from its starting position to the red agent's position while avoiding obstacles and managing its power efficiently.

Game Constraints

Power Loss: The blue agent loses 1 power unit for each move.
Obstacle Penalty: The blue agent loses 10 power units if it hits an obstacle.

Termination: The game ends if the blue agent's power reaches 0 or it successfully reaches the red agent.

Q-learning and Reinforcement Learning Implementation

State Representation:
The state is represented by the blue agent's position on the grid (x, y) and the remaining power.

Action Space:
The possible actions for the blue agent are: up, down, left, right.

Reward Function:
Positive reward (+100) for reaching the red agent.

Negative reward (-10) for hitting an obstacle.

Small penalty (-1) for moving to a new grid cell.

Penalty:
Large penalty for hitting an obstacle.
Small penalty for each move to discourage unnecessary wandering.

Implementation Details
State Space: (grid_x, grid_y, power)
Action Space: 4 actions (up, down, left, right)
Q-table Dimensions: (grid_size, grid_size, initial_power + 1, number_of_actions)
Training and Simulation
Training Parameters:

Learning Rate (alpha): 0.1
Discount Factor (gamma): 0.99
Exploration Factor (epsilon): 1.0, with decay
Number of Episodes: 500,000
Policy Extraction:

Determines the best action for each state based on the learned Q-table.
Simulation:

Uses the learned policy to navigate the blue agent from start to goal, visualizing the path taken and obstacles encountered.
Visualization
Red Agent: Marked in red.
Blue Agent Path: Shown in blue.
Obstacles: Indicated in black.

