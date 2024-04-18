
# TASK 1
import pygame
import random
import math

# Initialise Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FIELD = pygame.Rect(50, 50, WIDTH-100, HEIGHT-100)
ROBOT_RADIUS = 20
WHEEL_RADIUS = 5
TARGET_RADIUS = 10
FONT = pygame.font.SysFont("Arial", 24)

def new_episode(episode = -1):
    robot_pose = [random.randint(FIELD.left, FIELD.right), 
                  random.randint(FIELD.top, FIELD.bottom),
                  random.randint(0,359)]    
    target_pos = [random.randint(FIELD.left, FIELD.right), 
                  random.randint(FIELD.top, FIELD.bottom)]
    
    return robot_pose, target_pos, episode + 1, 0


def clip(value, min_val = -1, max_val = 1):
    return max(min(value, max_val), min_val)


def update_pose(x, y, theta, omega_0, omega_1, omega_2, step_size=1.0):
    omega_0 = clip(omega_0)
    omega_1 = clip(omega_1)
    omega_2 = clip(omega_2)
    
    R = 0.5
    # d = 1.0
    V_x = R * (omega_0 * math.cos(math.radians(60)) +
               omega_1 * math.cos(math.radians(180)) +
               omega_2 * math.cos(math.radians(300)))
    V_y = R * (omega_0 * math.sin(math.radians(60)) +
               omega_1 * math.sin(math.radians(180)) +
               omega_2 * math.sin(math.radians(300)))
    V_x_rotated = (V_x * math.cos(math.radians(theta)) - 
                   V_y * math.sin(math.radians(theta)))
    V_y_rotated = (V_x * math.sin(math.radians(theta)) + 
                   V_y * math.cos(math.radians(theta)))

    omega = omega_0 + omega_1 + omega_2
    x_prime = x + V_x_rotated * step_size
    y_prime = y + V_y_rotated * step_size

    theta_prime = theta + omega * step_size
    theta_prime = theta_prime % 360
    return x_prime, y_prime, theta_prime

def compute_state(robot_x, robot_y, robot_theta, target_x, target_y):
    # Find the relative position of the target
    rel_x = target_x - robot_x
    rel_y = target_y - robot_y

    # Angle from robot to target
    angle_to_target = math.degrees(math.atan2(rel_y, rel_x)) % 360

    # Adjusting this angle based on the robot's orientation
    relative_angle = (angle_to_target - robot_theta + 360) % 360

    # Determine state based on relative_angle
    if 0 <= relative_angle < 60:
        return 0  # ball in front of the robot
    elif 60 <= relative_angle < 120:
        return 1  # ball in front right of the robot
    elif 120 <= relative_angle < 240:
        return 3  # ball right in behind the robot
    elif 240 <= relative_angle < 300:
        return 4  # ball left in behind the robot
    else:
        return 2  # ball in front left of the robot

# Generate all possible combinations for three wheels
wheel_actions = [-0.5, 0.0, 0.5]
possible_actions = [(w0, w1, w2) for w0 in wheel_actions for w1 in wheel_actions for w2 in wheel_actions]
##

# Start first episode
score = 0
running = True
omega_0, omega_1, omega_2 = 0, 0, 0

# Initialize Q-table with zeros
number_of_states = 5
number_of_actions = len(possible_actions)
Q = [[0 for a in range(number_of_actions)] for s in range(number_of_states)]

# Hyperparameters
alpha = 1 # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

reward_history = []  # To store the total reward per episode

[x,y,theta], target_pos, episode, step = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    #Replacement of random action generation with specified action generation
    current_state = compute_state(x, y, theta, target_pos[0], target_pos[1])

    # Select a random action from the possible actions
    # Epsilon-greedy action selection
    if random.uniform(0, 1) < epsilon:
        action = random.choice(possible_actions)
        # print(action)
    else:
        action_idx = Q[current_state].index(max(Q[current_state]))
        action = possible_actions[action_idx]

    omega_0, omega_1, omega_2 = action

    x, y, theta = update_pose(x, y, theta, omega_0, omega_1, omega_2) # computing new position of robot using current positions
    next_state = compute_state(x, y, theta, target_pos[0], target_pos[1]) # updating to new state

    step += 1
    score -= 0.1
    reward = -0.1
    # Check for target, timeout, or out-of-bounds
    distance_to_target = math.sqrt((x - target_pos[0])**2 + (y - target_pos[1])**2)
    if distance_to_target <= ROBOT_RADIUS:
        reward += 10
    elif not FIELD.collidepoint(x, y):
        reward -= 10

    # Q value update
    action_idx = possible_actions.index(action)
    best_next_action = max(Q[next_state])
    Q[current_state][action_idx] += alpha * (reward + gamma * best_next_action - Q[current_state][action_idx])

    # Update epsilon for epsilon-greedy policy
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Store cumulative reward for the episode
    if not reward_history or step == 1:
        reward_history.append(reward)
    else:
        reward_history[-1] += reward

    score += reward  # Using the reward to update the score

    # Start a new episode if conditions met
    if distance_to_target <= ROBOT_RADIUS or not FIELD.collidepoint(x, y) or step > 1000:
        [x, y, theta], target_pos, episode, step = new_episode(episode)

    # Draw everything
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (25, 25, 25), FIELD)

    pygame.draw.circle(screen, (200, 200, 200), (int(x), HEIGHT - int(y)), ROBOT_RADIUS)
    pygame.draw.circle(screen, (255, 165, 0), (int(target_pos[0]), HEIGHT - int(target_pos[1])), TARGET_RADIUS)
    
    # Draw Wheels
    for i, colour in zip([60, 180, 300], [(255, 0, 0), (255, 0, 255), (0, 0, 255)]):
        wheel_x = int(x + ROBOT_RADIUS * math.cos(math.radians(i + theta - 90)))
        wheel_y = HEIGHT - int(y - ROBOT_RADIUS * math.sin(math.radians(i + theta - 90)))
        pygame.draw.circle(screen, colour, (wheel_x, wheel_y), WHEEL_RADIUS)
    
    score_surface = FONT.render(f'Episode: {episode}  Step: {step}  Score: {score:.2f}', True, (255, 255, 255))
    screen.blit(score_surface, (WIDTH - 400, 10))
    
    pygame.display.flip()
    pygame.time.delay(50)

import matplotlib.pyplot as plt

plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.show()

# Once the running loop is finished, print the Q-table
print("\nQ-table at the end of the simulation:")
for state_idx, state_actions in enumerate(Q):
    print(f"State {state_idx}: {state_actions}")

# # Testing
# import numpy as np
#
#
# def test_agent(Q, epsilon=0.0, num_episodes=50):
#     rewards_per_episode = []
#
#     for episode in range(num_episodes):
#         [x, y, theta], target_pos, _, step = new_episode()
#         total_reward = 0
#
#         while True:
#             current_state = compute_state(x, y, theta, target_pos[0], target_pos[1])
#
#             action_idx = Q[current_state].index(max(Q[current_state]))
#             action = possible_actions[action_idx]
#             # Epsilon-greedy action selection
#             # if random.uniform(0, 1) < epsilon:
#             #     action = random.choice(possible_actions)
#             # else:
#             #     action_idx = Q[current_state].index(max(Q[current_state]))
#             #     action = possible_actions[action_idx]
#
#             omega_0, omega_1, omega_2 = action
#             x, y, theta = update_pose(x, y, theta, omega_0, omega_1, omega_2)
#             reward = -0.01
#
#             distance_to_target = math.sqrt((x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2)
#             if distance_to_target <= ROBOT_RADIUS:
#                 reward += 10 # reward for reaching the target
#                 break
#
#             elif not FIELD.collidepoint(x, y):
#                 reward -= 10
#                 break
#
#             total_reward += reward
#
#             if step > 1000:
#                 break
#
#             step += 1 # increment the step counter
#
#         rewards_per_episode.append(total_reward)
#
#     return rewards_per_episode
#
#
# # Running tests
# epsilon_for_test = 0.0
#
# # Using the trained policy
# rewards_new_policy = test_agent(Q, epsilon=epsilon_for_test)
# print("Test Avg for New Policy:", np.mean(rewards_new_policy))
# print("Test StdDev for New Policy:", np.std(rewards_new_policy))
#
# # Using the initial policy (with a Q table initialized with zeros)
# Q_initial = [[0 for a in range(number_of_actions)] for s in range(number_of_states)]
# rewards_initial_policy = test_agent(Q_initial, epsilon=epsilon_for_test)
# print("Test Avg for Initial Policy:", np.mean(rewards_initial_policy))
# print("Test StdDev for Initial Policy:", np.std(rewards_initial_policy))
#
