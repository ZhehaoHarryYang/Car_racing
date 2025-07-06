import gymnasium as gym
from gymnasium import spaces
import numpy as np
import track  # Import the track module for boundary calculations

class SimpleCarRacingEnv(gym.Env):
    def __init__(self):
        super(SimpleCarRacingEnv, self).__init__()
        
        # Define the action space: 0 = turn left, 1 = turn right
        self.action_space = spaces.Discrete(2)
        
        # Define the observation space: (x_position, y_position, angle)
        self.observation_space = spaces.Box(low=np.array([0, -track.TRACK_WIDTH / 2, -np.pi]), 
                                            high=np.array([track.TRACK_LENGTH, track.TRACK_WIDTH / 2+15, np.pi]), 
                                            dtype=np.float32)

        # Initial car settings
        self.speed = 1.0  # Initial speed
        self.car_position = np.array([0.0, 0.0])  # Start at the center
        self.angle = 0  # Facing right
        self.done = False
        self.finish_line = track.TRACK_LENGTH  # Finish line is at the end of the track

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_position = np.array([0.0, 0.0])  # Reset to the start of the track center
        self.angle = 0
        self.speed = 1.0  # Reset speed
        self.done = False
        observation = np.array([self.car_position[0], self.car_position[1], self.angle])
        return observation, {}

    def step(self, action):
        # Save the previous position to compare with the new position
        prev_x_position = self.car_position[0]

        # Update angle based on action: 0 = turn left, 1 = turn right
        turn_angle = np.pi / 36  # 5 degrees in radians
        if action == 0:  # Turn left
            self.angle += turn_angle
        elif action == 1:  # Turn right
            self.angle -= turn_angle

        # **Wrap angle between 0 and 2π (360 degrees)**
        self.angle %= 2 * np.pi

        # Predict the next position based on the current angle and speed
        predicted_position = np.copy(self.car_position)
        predicted_position[0] += np.cos(self.angle) * self.speed
        predicted_position[1] += np.sin(self.angle) * self.speed

        # Get current track boundaries for the predicted position
        left_boundary = track.left_boundary(predicted_position[0])
        right_boundary = track.right_boundary(predicted_position[0])

        # Default reward for staying on track
        reward = 1

        # Positive reward for moving forward (positive x direction)
        if predicted_position[0] > prev_x_position:
            reward += 2  # Reward for moving forward

        # Penalty for moving backward (negative x direction)
        if predicted_position[0] < prev_x_position:
            reward -= 3  # Penalty for moving backward

        # Check if the predicted position is off the track
        if (predicted_position[1] < left_boundary or predicted_position[1] > right_boundary):
            # If the predicted position is off the track, place the car at the nearest boundary
            if predicted_position[1] < left_boundary:
                self.car_position[1] = left_boundary
            elif predicted_position[1] > right_boundary:
                self.car_position[1] = right_boundary

            # Set the speed to 0 immediately on hitting the boundary
            self.speed = 0

            reward = -10  # Penalty for going off track

        elif self.car_position[0] >= self.finish_line:
            # If the car reaches the finish line, reward more
            reward = 100  # Large reward for completing the track
            self.done = True  # End the episode only when track is finished

        else:
            # Gradually restore speed when on track
            self.speed = min(self.speed + 0.1, 1.0)  # Increase speed up to max

        # Update the car position after the action (if it's still on the track or adjusted to boundary)
        self.car_position[0] += np.cos(self.angle) * self.speed
        self.car_position[1] += np.sin(self.angle) * self.speed

        truncated = False  # No truncation logic
        observation = np.array([self.car_position[0], self.car_position[1], self.angle])
        return observation, reward, self.done, truncated, {}

    def render(self):
        # Display the car's position, angle, speed, and track boundaries at the current x position
        left_boundary = track.left_boundary(self.car_position[0])
        right_boundary = track.right_boundary(self.car_position[0])
        print(f"Car Position: {self.car_position}, Angle: {self.angle * 180 / np.pi:.2f} degrees, Speed: {self.speed:.2f}")
        print(f"Track Boundaries at x={self.car_position[0]:.2f}: Left={left_boundary:.2f}, Right={right_boundary:.2f}")

    def close(self):
        pass
