import math
import random

from simple_dqn_tf2 import Agent
from simple_dqn_tf2 import plot_learning

import numpy as np
import pygame
import tensorflow as tf

from collections import deque as queue

pygame.init()

d_row = [-1, 0, 1, 0]
d_col = [0, 1, 0, -1]


def is_valid(vis, row, col, grid):
    if row < 0 or col < 0 or row > 9 or col > 9:
        return False

    if grid[row][col] == 0:
        return False

    if vis[row][col]:
        return False

    return True


def breath_first_search(grid, vis, row, col):
    q = queue()
    q.append((row, col))

    if is_valid(vis, row, col, grid):
        vis[row][col] = True

    c = 0

    while len(q) > 0:
        cell = q.popleft()
        x = cell[0]
        y = cell[1]

        c += 1

        for i in range(4):
            adj_x = x + d_row[i]
            adj_y = y + d_col[i]

            if is_valid(vis, adj_x, adj_y, grid):
                q.append((adj_x, adj_y))
                vis[adj_x][adj_y] = True

    return c


class Environment:
    def __init__(self):
        self.display = pygame.display.set_mode((800, 450))
        self.window = pygame.Surface((800, 450))
        self.clock = pygame.time.Clock()

        self.player = self.Player()
        self.food = self.Food()

        self.s = 0
        self.f = 0

        self.grid = [[.5 for _ in range(10)] for _ in range(10)]
        for p in self.player.positions:
            self.grid[p[1]][p[0]] = 0

        self.vis = [[False for _ in range(10)] for _ in range(10)]
        p = self.player.positions[0]

        self.observation = [abs(p[0] - self.food.x - 1) + abs(p[1] - self.food.y),
                            abs(p[0] - self.food.x) + abs(p[1] - self.food.y + 1),
                            abs(p[0] - self.food.x + 1) + abs(p[1] - self.food.y),
                            abs(p[0] - self.food.x) + abs(p[1] - self.food.y - 1),
                            breath_first_search(self.grid, self.vis, p[0] - 1, p[1]),
                            breath_first_search(self.grid, self.vis, p[0], p[1] + 1),
                            breath_first_search(self.grid, self.vis, p[0] + 1, p[1]),
                            breath_first_search(self.grid, self.vis, p[0], p[1] - 1),
                            len(self.player.positions)
                            ]

        self.reward = 0
        self.done = 0
        self.info = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        self.s += 1

    def reset(self):
        self.__init__()
        return self.observation

    def render(self):
        self.clock.tick()
        self.window.fill((255, 255, 255))

        for x in range(10):
            for y in range(10):
                pygame.draw.rect(self.window, (0, 0, 0), (x * 20, y * 20, 20, 20), 2)

        self.player.blit(self.window)
        self.food.blit(self.window)

        self.handle_events()
        self.display.blit(self.window, (0, 0))
        pygame.display.update()

    def step(self, action):
        self.player.move(action)

        p = self.player.positions[0]

        if p[0] < 0 or p[0] >= 10 or p[1] < 0 or p[1] >= 10 or self.player.positions.count(p) > 1 or self.f > 100:
            self.done = 1
            self.reward -= 1000

        self.reward = pow(self.player.fitness * 2, 2) * pow(self.s, 1.5)

        if p == [self.food.x, self.food.y]:
            self.player.fitness += 1
            self.player.positions.append(p)
            self.food.reset()
            self.f = 0

        self.grid = [[.5 for _ in range(10)] for _ in range(10)]
        self.grid[self.food.y][self.food.x] = 1

        for p in self.player.positions:
            if 0 <= p[0] < 10 and 0 <= p[1] < 10:
                self.grid[p[1]][p[0]] = 0

        self.f += 1
        self.s += 1

        self.observation = [abs(p[0] - self.food.x - 1) + abs(p[1] - self.food.y),
                            abs(p[0] - self.food.x) + abs(p[1] - self.food.y + 1),
                            abs(p[0] - self.food.x + 1) + abs(p[1] - self.food.y),
                            abs(p[0] - self.food.x) + abs(p[1] - self.food.y - 1),
                            breath_first_search(self.grid, self.vis, p[0] - 1, p[1]),
                            breath_first_search(self.grid, self.vis, p[0], p[1] + 1),
                            breath_first_search(self.grid, self.vis, p[0] + 1, p[1]),
                            breath_first_search(self.grid, self.vis, p[0], p[1] - 1),
                            len(self.player.positions)
                            ]

        return self.observation, self.reward, self.done, self.info

    class Player:
        def __init__(self):
            self.d = 0
            self.positions = [[5, 5 - i] for i in range(3)]
            self.fitness = 0

        def blit(self, surface):
            for position in self.positions:
                pygame.draw.rect(surface, (0, 0, 200), (position[0] * 20, position[1] * 20, 20, 20))

        def move(self, action):
            x = self.positions[0][0]
            y = self.positions[0][1]

            if action == 0:
                x += 1

            if action == 1:
                x -= 1

            if action == 2:
                y += 1

            if action == 3:
                y -= 1

            self.positions.insert(0, [x, y])
            del self.positions[-1]

    class Food:
        def __init__(self):
            self.x = random.randint(0, 9)
            self.y = random.randint(0, 9)

        def reset(self):
            self.__init__()

        def blit(self, surface):
            pygame.draw.rect(surface, (200, 0, 0), (self.x * 20, self.y * 20, 20, 20))


def run():
    lr = 0.001
    n_games = 2000

    env = Environment()
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr,
                  input_dims=(9,),
                  n_actions=4, mem_size=1000000, batch_size=64,
                  epsilon_end=0.01)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            env.render()

            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)

            score += reward
            agent.store_transition(observation, action, reward, new_observation, done)
            observation = new_observation
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("episode: ", i, "score %.2f" % score,
              "average_score %.2f" % avg_score,
              "epsilon %.2f" % agent.epsilon)

    filename = "snake.png"
    x = [i + 1 for i in range(n_games)]
    plot_learning(x, scores, eps_history, filename)
    agent.save_model()
    print(agent.q_eval.summary())


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    run()
