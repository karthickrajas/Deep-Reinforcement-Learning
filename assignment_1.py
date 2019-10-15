# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:23:01 2019

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

n_rows = 5
n_cols = 5

rewards = np.zeros((n_rows,n_cols))
rewards[0,1] = 10
rewards[0,3] = 5

lamda = 0.95
gamma = 0.5

actions = np.array(['up','down','left','right'])

loc = (0,0)

achieved = []
chosen_actions = []
cum_achieved = [0]

state_value = np.zeros((n_rows,n_cols))
eligibility_value = np.zeros((n_rows,n_cols))

def get_actions():
    return np.random.choice(actions)

def get_state_reward(loc, action):
    x, y = loc
    reward = 0
    if action=="up":
        if x-1 < 0:
            x = x
            y = y
            reward = -1
        else:
            x = x-1
            y = y
    if action=="down":
        if x+1 >= n_rows:
            x = x
            y = y
            reward = -1
        else:
            x = x+1
            y = y
    if action=="left":
        if y-1 < 0:
            x = x
            y = y
            reward = -1
        else:
            x = x
            y = y-1
    if action=="right":
        if y+1 < 0:
            x = x
            y = y
            reward = -1
        else:
            x = x
            y = y+1
    
    if x==0 and y==1:
        reward = rewards[(x,y)]
        x = 4
        y = y
    
    if x==0 and y==3:
        reward = rewards[(x,y)]
        x = 4
        y = y
    
    return (x,y), reward


for _ in range(100000):
    chosen_action = get_actions()
    chosen_actions.append(chosen_action)
    new_loc, reward = get_state_reward(loc,chosen_action)
    eligibility_value = eligibility_value *lamda * gamma
    eligibility_value[loc] = eligibility_value[loc] + 1
    td_error = reward + gamma * state_value[new_loc] - state_value[loc]
    state_value = state_value + 1 * td_error * eligibility_value
    achieved.append(reward)
    cum_achieved.append(achieved[-1]+cum_achieved[-1])

plt.plot(cum_achieved)
plt.show()

plt.hist(chosen_actions)
plt.show()
   