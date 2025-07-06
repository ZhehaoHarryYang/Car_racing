from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from CarEnvCustom import SimpleCarRacingEnv

env = SimpleCarRacingEnv()

# Wrap the environment with DummyVecEnv for vectorization
env = DummyVecEnv([lambda: env])  # Ensure compatibility with stable-baselines3

model_path = "car_race_model9.zip"

try:
    # Load the existing model
    model = PPO.load(model_path, env=env)
    print("Loaded existing model for further training.")
except FileNotFoundError:
    # Initialize a new model if no saved model exists
    model = PPO("MlpPolicy", env, learning_rate=0.0003, batch_size=64, gamma=0.99, verbose=1, ent_coef=0.01)
    print("No pre-trained model found. Starting new training.")

# Train the model
additional_timesteps = 20000  # Set the number of additional timesteps to train

model.learn(total_timesteps=additional_timesteps)

# Save the model
model.save(model_path)

env.close()
