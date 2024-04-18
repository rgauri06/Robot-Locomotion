# TASK 2

import pygame
import random
import math
import numpy as np
import matplotlib.pyplot as plt

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
    target_vel = [random.uniform(0.2, 0.6),
                  random.uniform(-0.6, 0.6)]
    
    return robot_pose, target_pos, target_vel, episode + 1, 0


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
    V_x_rotated = (V_x * math.cos(math.radians(-theta)) - 
                   V_y * math.sin(math.radians(-theta)))
    V_y_rotated = (V_x * math.sin(math.radians(-theta)) + 
                   V_y * math.cos(math.radians(-theta)))

    omega = omega_0 + omega_1 + omega_2
    x_prime = x + V_x_rotated * step_size
    y_prime = y + V_y_rotated * step_size

    theta_prime = theta + omega * step_size
    theta_prime = theta_prime % 360
    return x_prime, y_prime, theta_prime

def compute_state(robot_x, robot_y, robot_theta, target_x, target_y, target_vel_x, target_vel_y):
    # Existing relative position calculations
    rel_x = target_x - robot_x
    rel_y = target_y - robot_y

    # Angle from robot to target
    angle_to_target = math.degrees(math.atan2(rel_y, rel_x)) % 360

    # Adjusting this angle based on the robot's orientation
    relative_angle = (angle_to_target - robot_theta + 360) % 360

    # Determine state based on relative_angle
    if 0 <= relative_angle < 60:
        angle_state = 0  # ball in front of the robot
    elif 60 <= relative_angle < 120:
        angle_state = 1  # ball in front right of the robot
    elif 120 <= relative_angle < 240:
        angle_state = 3  # ball right in behind the robot
    elif 240 <= relative_angle < 300:
        angle_state = 4  # ball left in behind the robot
    else:
        angle_state = 2  # ball in front left of the robot

    # Determine target velocity state
    if target_vel_x > 0:
        vel_x_state = 0  # moving right
    elif target_vel_x < 0:
        vel_x_state = 1  # moving left
    else:
        vel_x_state = 2  # stationary in x

    if target_vel_y > 0:
        vel_y_state = 0  # moving up
    elif target_vel_y < 0:
        vel_y_state = 1  # moving down
    else:
        vel_y_state = 2  # stationary in y

    # Combining the states for simplicity.
    # Assuming 5 angle states, 3 x velocity states, and 3 y velocity states,
    # We get a total of 5 x 3 x 3 = 45 states.
    state = angle_state + (vel_x_state * 5) + (vel_y_state * 5 * 3)

    return state

# Reducing actions
# Possible actions for a single wheel
wheel_actions = [-0.5, 0.0, 0.5]

# Generate all possible combinations for three wheels
possible_actions = [(w0, w1, w2) for w0 in wheel_actions for w1 in wheel_actions for w2 in wheel_actions]

# Start first episode
score = 0
running = True
omega_0, omega_1, omega_2 = 0, 0, 0

# Initialize Q-table: 45 states x 27 actions
Q = np.zeros((45, 27))

# Parameters for Q-learning
alpha = 2  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

# Start first episode
episode_rewards = []  # Store rewards for each episode

[x, y, theta], target_pos, target_vel, episode, step = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # Get current state
    current_state = compute_state(x, y, theta, target_pos[0], target_pos[1], target_vel[0], target_vel[1])

    # Choose action based on epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        action_idx = np.random.choice(27)
    else:
        action_idx = np.argmax(Q[current_state, :])

    action = possible_actions[action_idx]
    omega_0, omega_1, omega_2 = action

    # Update robot's position
    x, y, theta = update_pose(x, y, theta, omega_0, omega_1, omega_2)

    step += 1
    immediate_reward = -0.01
    score += immediate_reward

    # Update target position
    target_pos[0] += target_vel[0]
    target_pos[1] += target_vel[1]

    # Bounce off the sides
    if target_pos[0] <= 50 or target_pos[0] >= WIDTH - 50:
        target_vel[0] = -target_vel[0]
    if target_pos[1] <= 50 or target_pos[1] >= HEIGHT - 50:
        target_vel[1] = -target_vel[1]

    # Check for target, timeout, or out-of-bounds
    distance_to_target = math.sqrt((x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2)
    if distance_to_target <= ROBOT_RADIUS:
        immediate_reward += 10
        episode_rewards.append(score)
        [x, y, theta], target_pos, target_vel, episode, step = new_episode(episode)
    elif not FIELD.collidepoint(x, y):
        immediate_reward -= 10
        episode_rewards.append(score)
        [x, y, theta], target_pos, target_vel, episode, step = new_episode(episode)
    elif step > 1000:
        episode_rewards.append(score)
        [x, y, theta], target_pos, target_vel, episode, step = new_episode(episode)

    # Update Q-table using Q-learning formula
    new_state = compute_state(x, y, theta, target_pos[0], target_pos[1], target_vel[0], target_vel[1])
    Q[current_state, action_idx] = Q[current_state, action_idx] + alpha * (immediate_reward + gamma * np.max(Q[new_state, :]) - Q[current_state, action_idx])

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Drawings
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (25, 25, 25), FIELD)
    pygame.draw.circle(screen, (200, 200, 200), (int(x), HEIGHT - int(y)), ROBOT_RADIUS)
    pygame.draw.circle(screen, (255, 165, 0), (int(target_pos[0]), HEIGHT - int(target_pos[1])), TARGET_RADIUS)

    # Draw wheels
    for i, colour in zip([60, 180, 300], [(255, 0, 0), (255, 0, 255), (0, 0, 255)]):
        wheel_x = int(x + ROBOT_RADIUS * math.cos(math.radians(i + theta - 90)))
        wheel_y = HEIGHT - int(y - ROBOT_RADIUS * math.sin(math.radians(i + theta - 90)))
        pygame.draw.circle(screen, colour, (wheel_x, wheel_y), WHEEL_RADIUS)

    score_surface = FONT.render(f'Episode: {episode}  Step: {step}  Score: {score:.2f}', True, (255, 255, 255))
    screen.blit(score_surface, (WIDTH - 400, 10))
    pygame.display.flip()
    pygame.time.delay(50)

# Once all episodes are over, you can plot the rewards to see how well the agent performed
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Reward Over Time')
plt.show()
##
# Print the Q-table
np.set_printoptions(precision=2)  # Optional: Set the precision for better readability
print("\nQ-table:")
print(Q)
