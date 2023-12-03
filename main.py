import gymnasium as gym
import numpy as np
from actorCritic import ActorCritic
from utils import plot_learning_curve
from datetime import datetime
import atexit
import sys

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = ActorCritic(alpha=1e-4, gamma=0.95, action_space=env.action_space.n, a1=86, a2=64, c1=88, c2=32 )
    n_games = 1500

    filePath = "tmp/img/LunarLander"+"-"+str(datetime.now())+"-g"+str(agent.gamma)+"-a"+str(agent.alpha)+"a1"+str(agent.aD1_dims)+"a2"+str(agent.aD2_dims)+"c1"+str(agent.cD1_dims)+"c2"+str(agent.cD2_dims)+".png"
    figureFile = filePath
    highScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = False

    if loadCheckpoint:
        agent.load()
    for i in range (n_games):
        print(i)
        observation = env.reset()[0]
        done = False
        score = 0
        try:
            while not done:
                action, actionProba, probs = agent.chooseAction(observation)
                observation1, reward, done, _, info = env.step(action)
                score += reward
                if not loadCheckpoint:
                    agent.learn(observation, reward, observation1, done)
                observation = observation1
        except (KeyboardInterrupt, AssertionError, ValueError):
            print(action)
            print(actionProba.sample())
            print(probs)
            x = [j+1 for j in range (i)]
            plot_learning_curve(x, scoreHistory, figureFile)
            sys.exit()

        scoreHistory.append(score)
        averageScore = np.mean(scoreHistory[-100:])

        if averageScore > highScore:
            highScore = averageScore
            if not loadCheckpoint:
                agent.save()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % averageScore)

    x = [i+1 for i in range (n_games)]
    plot_learning_curve(x, scoreHistory, figureFile)
