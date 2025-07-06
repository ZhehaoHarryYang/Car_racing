import pygame
import numpy as np
import time  # Import time for delay
from CarRacingEnv import SimpleCarRacingEnv  # Import your custom environment
from stable_baselines3 import PPO  # Assuming you used PPO for training

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Racing Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (173, 216, 230)  # Light blue for the track

# Scaling factors for the car position and track bounds
track_width, track_height = 100, 5  # Logical dimensions of the track in your environment
scale = screen_width / track_width  # Scale for both x and y (800 / 100)

# Initialize environment
env = SimpleCarRacingEnv()
obs, _ = env.reset()  # Reset the environment and unpack the observation

# Load the trained model
model_path = "car_race_model3.zip"  # Replace with your model file path
model = PPO.load(model_path)

# Center the track on the screen
track_x_offset = (screen_width - track_width * scale) / 2
track_y_offset = (screen_height - track_height * scale) / 2

# Game loop
running = True
clock = pygame.time.Clock()
total_reward = 0  # Accumulated reward for the current episode
font = pygame.font.Font(None, 30)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Use the model to predict the action based on the current observation
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)  # Updated to match gymnasium API
    total_reward += reward  # Accumulate reward

    # Clear the screen
    screen.fill(WHITE)

    # Draw the track (background of the track)
    pygame.draw.rect(screen, BLUE, 
                     (track_x_offset + env.track_bounds["left"] * scale, 
                      track_y_offset + (track_height - env.track_bounds["top"]) * scale,  
                      track_width * scale, 
                      track_height * scale))

    # Draw track boundaries
    pygame.draw.rect(screen, BLACK, 
                     (track_x_offset + env.track_bounds["left"] * scale, 
                      track_y_offset + (track_height - env.track_bounds["top"]) * scale, 
                      track_width * scale, 
                      track_height * scale), 
                     2)

    # Calculate the car’s position on the screen (centered on the track)
    car_x = int(track_x_offset + env.car_position[0] * scale)
    car_y = screen_height - int(track_y_offset + env.car_position[1] * scale)  

    # Draw the car as a small red circle
    pygame.draw.circle(screen, RED, (car_x, car_y), 5)

    # Draw the car’s direction as a line
    direction_x = car_x + int(np.cos(env.angle) * 10)
    direction_y = car_y - int(np.sin(env.angle) * 10)
    pygame.draw.line(screen, BLACK, (car_x, car_y), (direction_x, direction_y), 2)

    # Display text information (speed, angle, current reward, accumulated reward)
    speed_text = font.render(f"Speed: {env.speed:.2f}", True, BLACK)
    angle_text = font.render(f"Angle: {np.degrees(env.angle):.2f}", True, BLACK)
    reward_text = font.render(f"Reward: {reward}", True, BLACK)
    total_reward_text = font.render(f"Total Reward: {total_reward}", True, BLACK)
    
    screen.blit(speed_text, (10, 10))
    screen.blit(angle_text, (10, 40))
    screen.blit(reward_text, (10, 70))
    screen.blit(total_reward_text, (10, 100))

    # Check if the episode is done
    if done or truncated:
        # Display end-of-episode message and total reward
        end_text = font.render("Episode finished! Press R to restart or Q to quit.", True, BLACK)
        screen.blit(end_text, (screen_width // 2 - 200, screen_height // 2))
        pygame.display.flip()
        
        # Wait for the player to choose to restart or quit
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_input = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset if 'R' is pressed
                        obs, _ = env.reset()  # Reset environment and observation if player chooses to restart
                        total_reward = 0  # Reset accumulated reward
                        waiting_for_input = False
                    elif event.key == pygame.K_q:  # Quit if 'Q' is pressed
                        running = False
                        waiting_for_input = False

    # Update display
    pygame.display.flip()
    
    # Slow down the loop
    time.sleep(0.1)  # Add delay to slow down the simulation

    # Control the frame rate
    clock.tick(30)

# Quit Pygame
pygame.quit()
