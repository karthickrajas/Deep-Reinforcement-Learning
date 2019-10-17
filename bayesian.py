# -*- coding: utf-8 -*-
"""
@author: Karthick
"""

import random
import numpy as np
import math

def find_max(pta):
    """
    function to find the amx pta : tie breaking using random
    input: pta(list) probability of each arm
    output : chosen_arm (number)
    action: chosing the best arm with random tie breaking 
    """
    winner = np.argwhere(pta==np.amax(pta)).reshape(-1)
    winner = int(random.choice(winner))
    return winner


class Thompson:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.numbers_of_success = [0] * n_arms
        self.numbers_of_failure = [0] * n_arms
        self.rewards = []
        self.chosen_arms = []
        return

    def action(self, context = False):
        self.max_random = 0
        self.chosen_arm = 0
        for i in range(0, self.n_arms):
            random_beta = random.betavariate(self.numbers_of_success[i] + 1, self.numbers_of_failure[i] + 1)
            if random_beta >self.max_random:
                self.max_random = random_beta
                self.chosen_arm = i
        self.chosen_arms.append(self.chosen_arm)
        return self.chosen_arm
    
    def update(self, reward):
        reward = int(reward)
        self.rewards.append(reward)
        if reward == 10 or reward == 5:
            self.numbers_of_success[self.chosen_arm] = self.numbers_of_success[self.chosen_arm] + 1
        else:
            self.numbers_of_failure[self.chosen_arm] = self.numbers_of_failure[self.chosen_arm] + 1


class eGreedy:
    def __init__(self, n_arms, epsilon):
        """
        Implemented one of the famous and simple techniques of eGreedy for online learning
        Inputs : n_arms, epsilon (number(int),number(float))
        output: None
        Action: creates and initiates arms, number of times arms being used
        """
        self.n_arms =n_arms
        self.epsilon = epsilon # eploration probability
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.chosen_arms = []
        self.rewards = []
        return
    
    def action(self, context=False):
        """
        actions :Choosen the most observed arm from the past for probability 1-epsilon else choosen randomly
        input : None 
        output : choosen arm (number)
        """
        z = np.random.random()
        if z > self.epsilon:
            # Pick the best arm
            chosen_arm = find_max(self.values)
            self.chosen_arms.append(chosen_arm)
            return chosen_arm
        # Randomly pick any arm with prob 1 / len(self.counts)
        chosen_arm = np.random.randint(0, self.n_arms)
        self.chosen_arms.append(chosen_arm)
        return chosen_arm
    
    def update(self, reward):
        """
        input : reward (number)
        output : None 
        """
        reward = int(reward)
        self.rewards.append(reward)
        # Increment chosen arm's count by one
        self.counts[self.chosen_arms[-1]] += 1
        n = self.counts[self.chosen_arms[-1]]
        # Recompute the estimated value of chosen arm using new reward
        value = self.values[self.chosen_arms[-1]]
        new_value = value * ((n - 1) / n) + reward / n
        self.values[self.chosen_arms[-1]] = new_value


class UCB:
    """
    Upper Confidence Bound (UCB) multi-armed bandit
    Arguments
    =========
    narms : int
        number of arms
    rho : float
        positive real explore-exploit parameter
    """
    def __init__(self, n_arms, rho = 2, q0 = np.inf):
        self.n_arms =n_arms
        self.rho = rho
        self.q0 = q0
        self.numbers_of_selections = [0] * n_arms
        self.avg_list = [q0 for i in range(n_arms)]
        self.q_list = [q0 for i in range(n_arms)]
        self.sums_of_rewards = [0] * n_arms
        self.chosen_arms = []
        self.rewards = []
        #self.max_upper_bound = 0
        #self.upper_bound = [0] * n_arms
        #self.average_reward = [0] * n_arms
        #self.chosen_arm = 0
        self.n_users = 0
        #self.avg_reward = 0
        return
    
    def action(self):
        self.n_users = self.n_users + 1
        values = self.q_list
        next_arm_list = []
        for index, value in enumerate(values):
            if value==max(values):
                next_arm_list.append(index)
        next_arm = int(np.random.choice(next_arm_list))
        self.chosen_arms.append(next_arm)
        return next_arm
        
    def update(self,reward):
        reward = int(reward)
        self.rewards.append(reward)
        index = self.chosen_arms[-1]
        average = self.avg_list[index]
        if average == np.inf:
            average = 0
        self.numbers_of_selections[index] = self.numbers_of_selections[index] + 1
        current = self.numbers_of_selections[index]
        # update the average value with previous weighted average value + new reward 
        self.avg_list[index] = (average * (current - 1) + reward) / current
        self.rewards.append(reward)
        self.numbers_of_selections[index] = self.numbers_of_selections[index] + 1
        self.sums_of_rewards[index] = self.sums_of_rewards[index] + reward
        # update extimate Q value of each chosen arm
        for i in range(self.n_arms):
            mean = self.avg_list[i]
            if mean != np.inf:  # update Q value for the arms that have been explored
                self.q_list[i] = mean + np.sqrt((self.rho * np.log(self.n_users)) / self.numbers_of_selections[i])