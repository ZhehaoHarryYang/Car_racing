import gymnasium as gym
from gymnasium import spaces
import numpy as np
from track_Custom import Track  # Import the Track class from track_Custom

class SimpleCarRacingEnv(gym.Env):
    def __init__(self):
        super(SimpleCarRacingEnv, self).__init__()

        # Create an instance of the Track class
        self.track = Track()

        # Define the action space: 0 = turn left, 1 = turn right
        self.action_space = spaces.Discrete(2)

        # Define the observation space:
        # (x_position, y_position, angle, distance_to_left_boundary, distance_to_right_boundary)
        self.observation_space = spaces.Box(low=np.array([0, -self.track.get_width() / 2, -np.pi, 0, 0]),
                                            high=np.array([self.track.track_length, self.track.get_width() / 2 + 20, np.pi, self.track.get_width() / 2, self.track.get_width() / 2]),
                                            dtype=np.float32)

        # Initial car settings
        self.speed = 1.0  # Initial speed
        self.car_position = np.array([0.0, 0.0])  # Start at the center
        self.angle = 0  # Facing right
        self.done = False
        self.finish_line = self.track.track_length  # Finish line is at the end of the track

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_position = np.array([0.0, 0.0])  # Reset to the start of the track center
        self.angle = 0
        self.speed = 1.0  # Reset speed
        self.done = False
        # Calculate initial distance to the left and right boundaries
        left_boundary = self.track.get_left_boundary(self.car_position[0])
        right_boundary = self.track.get_right_boundary(self.car_position[0])
        distance_to_left_boundary = self.car_position[1] - left_boundary
        distance_to_right_boundary = right_boundary - self.car_position[1]
        observation = np.array([self.car_position[0], self.car_position[1], self.angle, 
                               distance_to_left_boundary, distance_to_right_boundary])
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
        left_boundary = self.track.get_left_boundary(predicted_position[0])
        right_boundary = self.track.get_right_boundary(predicted_position[0])

        # Calculate distances to left and right boundaries
        distance_to_left_boundary = predicted_position[1] - left_boundary
        distance_to_right_boundary = right_boundary - predicted_position[1]

        # Calculate the distance to the middle of the track
        track_middle = (left_boundary + right_boundary) / 2
        distance_to_middle = abs(predicted_position[1] - track_middle)

        # Reward for staying in the middle of the track (closer to the center is better)
        middle_reward = max(0, 1 - distance_to_middle / (self.track.get_width() / 2))

        # Default reward for staying on track
        reward = middle_reward

        # Positive reward for moving forward (positive x direction)
        if predicted_position[0] > prev_x_position:
            reward += 2  # Reward for moving forward

        # Penalty for moving backward (negative x direction)
        if predicted_position[0] < prev_x_position:
            reward -= 3  # Penalty for moving backward
        # print(left_boundary, right_boundary)
        # Check if the predicted position is off the track
        if (predicted_position[1] < left_boundary or predicted_position[1] > right_boundary):
            # If the predicted position is off the track, place the car at the nearest boundary
            if predicted_position[1] < left_boundary:
                self.car_position[1] = left_boundary
            elif predicted_position[1] > right_boundary:
                self.car_position[1] = right_boundary

            # Set the speed to 0 immediately on hitting the boundary
            self.speed = 0

            # Apply a penalty for going off track and prevent any other rewards
            reward = -10  # Penalty for going off track

        elif self.car_position[0] >= self.finish_line:
            self.done = True  # End the episode only when track is finished

        else:
            # Gradually recover speed, but only if not off track
            if self.speed < 1.0:
                self.speed = min(self.speed + 0.1, 1.0)
            self.car_position[0] += np.cos(self.angle) * self.speed
            self.car_position[1] += np.sin(self.angle) * self.speed

        truncated = False  # No truncation logic
        observation = np.array([self.car_position[0], self.car_position[1], self.angle, 
                            distance_to_left_boundary, distance_to_right_boundary])

        return observation, reward, self.done, truncated, {}


    def render(self):
        # Display the car's position, angle, speed, and track boundaries at the current x position
        left_boundary = self.track.get_left_boundary(self.car_position[0])
        right_boundary = self.track.get_right_boundary(self.car_position[0])
        print(f"Car Position: {self.car_position}, Angle: {self.angle * 180 / np.pi:.2f} degrees, Speed: {self.speed:.2f}")
        print(f"Track Boundaries at x={self.car_position[0]:.2f}: Left={left_boundary:.2f}, Right={right_boundary:.2f}")

    def close(self):
        pass


def test_car_racing_env():
    # Create the environment
    env = SimpleCarRacingEnv()

    # Reset the environment to start a new episode
    observation, _ = env.reset()

    # Test the environment with a few actions (turning left or right)
    actions = [1,1,1,1]*3  # Alternating left (0) and right (1)
    
    for action in actions:
        print(f"Action: {'Left' if action == 0 else 'Right'}")

        # Step through the environment with the chosen action
        observation, reward, done, truncated, info = env.step(action)

        # Print the result after each action
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print("-" * 40)

        # If the episode is done (reached the finish line or went off track), break
        if done:
            print("Episode finished!")
            break

    # Close the environment
    env.close()

# Run the test
# test_car_racing_env()
