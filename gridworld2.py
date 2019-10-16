# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:40:13 2019

@author: Lenovo
"""


import numpy as np


class gridworld:
    
    def __init__(self, tot_row=5, tot_col=5):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        self.state_matrix = np.zeros((tot_row,tot_col))
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]
        
    def setStateMatrix(self, state_matrix):
        '''Set the obstacles in the world.
        '''
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix
        
    def setPosition(self, index_row=None, index_col=None):
        ''' Set the position of the robot in a specific state.
        '''
        if(index_row is None or index_col is None): self.position = [np.random.randint(self.world_row), np.random.randint(self.world_col)]
        else: self.position = [index_row, index_col]
    
    def step(self, action):
        ''' One step in the world.

        [observation, reward, done = env.step(action)]
        The robot moves one step in the world based on the action given.
        The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        @return observation the position of the robot after the step
        @return reward the reward associated with the next state
        @return done True if the state is terminal  
        '''
        s_0 = self.position
        a_0 = action
        
        if(action >= self.action_space_size):
            raise ValueError('The action is not included in the action space.')

        #Generating a new position based on the current position and action
        if(action == 0): new_position = [self.position[0]-1, self.position[1]]   #UP
        elif(action == 1): new_position = [self.position[0], self.position[1]+1] #RIGHT
        elif(action == 2): new_position = [self.position[0]+1, self.position[1]] #DOWN
        elif(action == 3): new_position = [self.position[0], self.position[1]-1] #LEFT
        else: raise ValueError('The action is not included in the action space.')
        
        if (new_position[0]==0 and new_position[1]==1):
            reward = 30
            new_position[0]=3
            new_position[1]=1
            self.position = new_position
        
        elif (new_position[0]==0 and new_position[1]==3):
            reward = 10
            new_position[0]=2
            new_position[1]=3       
            self.position = new_position
            
        elif (new_position[0]<0 or new_position[0]>=self.world_row):
            reward = -1
            
        elif(new_position[1]<0 or new_position[1]>=self.world_col):
            reward = -1
            
        else:
            reward = 0
            self.position = new_position
        r_1 = reward
        s_1 = new_position
        
        return r_1, s_1