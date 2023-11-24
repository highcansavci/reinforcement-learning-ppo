import gym
import numpy as np
from agent.ppo_agent import Agent
from utils.utils import plot_learning_curve


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    env = gym.wrappers.RecordVideo(env, 'video')
    num_horizon = 20
    batch_size = 5
    n_epochs = 4
    alpha = 3e-4
    agent_ = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                   n_epochs=n_epochs, input_dims=env.observation_space.shape)
    agent_.load_models()
    n_games = 5

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent_.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = observation_