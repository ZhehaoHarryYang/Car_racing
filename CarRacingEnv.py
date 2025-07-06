import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleCarRacingEnv(gym.Env):
    def __init__(self):
        super(SimpleCarRacingEnv, self).__init__()
        
        # Define the action space: 0 = turn left, 1 = turn right
        self.action_space = spaces.Discrete(2)
        
        # Define the state space: (x_position, y_position, angle)
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi]), 
                                            high=np.array([100, 5, np.pi]), dtype=np.float32)

        # Track boundaries for a 5x100 track
        self.track_bounds = {
            "left": 0,
            "right": 100,
            "top": 5,
            "bottom": 0
        }
        
        # Initial car settings
        self.speed = 1.0  # Initial speed
        self.car_position = np.array([0.0, 2.5])  # Centered at the start of the track
        self.angle = np.pi / 4  # Start at 45 degrees to encourage learning turns
        self.done = False
        self.finish_line = self.track_bounds["right"]  # Define finish line at the end of the track

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_position = np.array([0.0, 2.5])  # Reset to the start of the track
        self.angle = np.pi / 4  # 45 degrees to the right
        self.speed = 1.0  # Reset speed
        self.done = False
        observation = np.array([self.car_position[0], self.car_position[1], self.angle])
        return observation, {}

    def step(self, action):
        # Update angle based on action: 0 = turn left, 1 = turn right
        turn_angle = np.pi / 18  # 10 degrees in radians
        if action == 0:  # Turn left
            self.angle += turn_angle
        elif action == 1:  # Turn right
            self.angle -= turn_angle

        # Update car position based on the current angle and speed
        self.car_position[0] += np.cos(self.angle) * self.speed
        self.car_position[1] += np.sin(self.angle) * self.speed

        # Default reward for staying on track
        reward = 1  

        # Check if the car hits the track boundary
        if (self.car_position[0] < self.track_bounds["left"] or 
            self.car_position[0] > self.track_bounds["right"] or 
            self.car_position[1] < self.track_bounds["bottom"] or 
            self.car_position[1] > self.track_bounds["top"]):

            # Calculate impact angle for boundary collision
            impact_angle = np.arctan2(
                self.car_position[1] - (self.track_bounds["top"] if self.car_position[1] > self.track_bounds["top"] else self.track_bounds["bottom"]),
                self.car_position[0] - (self.track_bounds["right"] if self.car_position[0] > self.track_bounds["right"] else self.track_bounds["left"])
            )
            speed_reduction = max(0, np.cos(impact_angle))  # Speed reduction on impact
            self.speed *= speed_reduction

            # Stop the car if speed drops very low
            if self.speed < 0.1:
                self.speed = 0

            reward = -10  # Penalty for hitting the boundary
            self.done = True  # End the episode

        elif self.car_position[0] >= self.finish_line:
            # If the car reaches the finish line, reward more
            reward = 100  # Large reward for completing the track
            self.done = True  # End the episode
        else:
            # Gradually restore speed when on track
            self.speed = min(self.speed + 0.05, 1.0)  # Increase speed up to max

        truncated = False  # No truncation logic
        observation = np.array([self.car_position[0], self.car_position[1], self.angle])
        return observation, reward, self.done, truncated, {}

    def render(self):
        # Simple render function (optional)
        print(f"Car Position: {self.car_position}, Angle: {self.angle * 180 / np.pi:.2f} degrees, Speed: {self.speed:.2f}")

    def close(self):
        pass


# env = SimpleCarRacingEnv()

# # Run a few test episodes
# for episode in range(3):  # Run 3 episodes
#     observation, info = env.reset()  # Reset environment at the start of each episode
#     print(f"Episode {episode + 1} Start - Car Position: {observation[:2]}, Angle: {observation[2] * 180 / np.pi:.2f} degrees")

#     done = False
#     while not done:
#         # Random action (turn left or turn right)
#         action = env.action_space.sample()
        
#         # Take action and observe the result
#         observation, reward, done, truncated, info = env.step(action)

#         # Print car's current position, angle, and reward
#         print(f"Car Position: {observation[:2]}, Angle: {observation[2] * 180 / np.pi:.2f} degrees, Reward: {reward}")
    
#     print(f"Episode {episode + 1} Finished\n")

# env.close()