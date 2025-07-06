import pygame
import numpy as np
import time
from CarEnvTurn import SimpleCarRacingEnv
from stable_baselines3 import PPO
import track  # Import track for consistent boundary calculations

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Curved Track Car Racing Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (173, 216, 230)
COLLISION_COLOR = (255, 69, 0)  # Orange for collision effect

# Environment and model
env = SimpleCarRacingEnv()
obs, _ = env.reset()
model_path = "car_race_model8.zip"  # Replace with your model file path
model = PPO.load(model_path)

# Track parameters
track_width = track.TRACK_WIDTH
track_length = track.TRACK_LENGTH  # Logical length of track

# Scaling factors for display
scale = screen_width / track_length

# Car parameters
car_x = 0  # Start in the middle of the screen
car_y = screen_height // 2
car_angle = 0  # Angle of the car (in radians)
car_speed = 1.0 * scale  # Initial speed of the car
normal_speed = 1.0 * scale  # Normal speed when not colliding
min_speed = 0  # Minimum speed on collision

# Collision control
collision = False
collision_start_time = 0
collision_duration = 500  # Duration to display collision effect in milliseconds

# Timer to track the time to finish
start_time = pygame.time.get_ticks()

# Helper function to invert y-coordinate
def invert_y(y):
    """Inverts the y-coordinate for Pygame's coordinate system."""
    return screen_height - y

# Game loop
running = True
clock = pygame.time.Clock()
total_reward = 0
font = pygame.font.Font(None, 30)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check if the car has reached the end of the track before checking for collisions
    done = False
    if car_x >= track_length * scale:  # Car reaches the end of the track
        done = True  # Mark the episode as done

    # Only check for collision if the car hasn't finished
    if not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, truncated, _ = env.step(action)

        car_angle = env.angle
        total_reward += reward

        # Update car position based on angle and speed
        car_x += int(np.cos(car_angle) * car_speed)
        car_y += int(np.sin(car_angle) * car_speed)

        # Check for collision and adjust as necessary
        next_car_x = car_x + int(np.cos(car_angle) * car_speed)
        next_car_y = car_y + int(np.sin(car_angle) * car_speed)

        # Calculate boundaries with offset
        track_offset = screen_height // 2
        next_left_boundary = track.left_boundary(next_car_x / scale) * scale + track_offset
        next_right_boundary = track.right_boundary(next_car_x / scale) * scale + track_offset

        # Handle collisions
        if next_car_y < next_left_boundary or next_car_y > next_right_boundary:
            car_speed = 0
            collision = True
            collision_start_time = pygame.time.get_ticks()

            # Clamp car_y to the closest boundary
            if next_car_y < next_left_boundary:
                car_y = next_left_boundary
            elif next_car_y > next_right_boundary:
                car_y = next_right_boundary
        else:
            collision = False
            car_speed = min(car_speed + 0.1 * scale, normal_speed)

    # Clear screen
    screen.fill(WHITE)

    # Draw track boundaries
    for x in range(0, track_length):
        left_x = x * scale
        left_y = invert_y(track.left_boundary(x) * scale + track_offset)
        right_y = invert_y(track.right_boundary(x) * scale + track_offset)
        pygame.draw.line(screen, BLUE, (left_x, left_y), (left_x, right_y), 8)

    # Draw the car
    car_color = COLLISION_COLOR if collision and pygame.time.get_ticks() - collision_start_time < collision_duration else RED
    pygame.draw.circle(screen, car_color, (car_x, invert_y(car_y)), 5)

    # Draw car’s direction line
    direction_length = 20
    direction_x = car_x + int(np.cos(car_angle) * direction_length)
    direction_y = car_y + int(np.sin(car_angle) * direction_length)
    pygame.draw.line(screen, BLACK, (car_x, invert_y(car_y)), (direction_x, invert_y(direction_y)), 2)

    # Display text information
    speed_text = font.render(f"Speed: {car_speed:.2f}", True, BLACK)
    angle_text = font.render(f"Angle: {np.degrees(car_angle):.2f}", True, BLACK)
    reward_text = font.render(f"Reward: {reward}", True, BLACK)
    total_reward_text = font.render(f"Total Reward: {total_reward}", True, BLACK)
    screen.blit(speed_text, (10, 10))
    screen.blit(angle_text, (10, 40))
    screen.blit(reward_text, (10, 70))
    screen.blit(total_reward_text, (10, 100))

    # Display collision message if there was a recent collision
    if not done and collision:
        collision_text = font.render("Collision!", True, COLLISION_COLOR)
        screen.blit(collision_text, (screen_width // 2 - 50, screen_height // 2 - 20))

    # End of episode prompt
    if done or truncated:
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Calculate time in seconds
        end_text = font.render(f"Episode finished! Time: {elapsed_time:.2f}s. Press R to restart or Q to quit.", True, BLACK)
        screen.blit(end_text, (screen_width // 2 - 200, screen_height // 2))
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_input = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset environment and car variables
                        obs, _ = env.reset()
                        total_reward = 0
                        car_x = 0
                        car_y = screen_height // 2
                        car_angle = 0
                        car_speed = normal_speed
                        collision = False
                        done = False
                        start_time = pygame.time.get_ticks()  # Reset start time for the new episode
                        waiting_for_input = False
                    elif event.key == pygame.K_q:
                        running = False
                        waiting_for_input = False

    # Update display
    pygame.display.flip()
    clock.tick(30)
    time.sleep(0.1)

# Quit Pygame
pygame.quit()
