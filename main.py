import gymnasium as gym
import numpy as np
from actorCritic import ActorCritic
from utils import plot_learning_curve
from datetime import datetime
import atexit

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = ActorCritic(alpha=1e-5, gamma=0.50, action_space=env.action_space.n)
    n_games = 1000

    filePath = "tmp/img/LunarLander-g"+str(agent.gamma)+"-a"+str(agent.alpha)+"-"+str(datetime.now())+".png"
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
                action = agent.chooseAction(observation)
                observation1, reward, done, _, info = env.step(action)
                score += reward
                if not loadCheckpoint:
                    agent.learn(observation, reward, observation1, done)
                observation = observation1
        except KeyboardInterrupt:
            x = [j+1 for j in range (i)]
            plot_learning_curve(x, scoreHistory, figureFile)
        
        scoreHistory.append(score)
        averageScore = np.mean(scoreHistory[-100:])
        
        if averageScore > highScore:
            highScore = averageScore
            if not loadCheckpoint:
                agent.save()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % averageScore)

    x = [i+1 for i in range (n_games)]
    plot_learning_curve(x, scoreHistory, figureFile)
