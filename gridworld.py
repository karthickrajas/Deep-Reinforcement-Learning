#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Class for creating a gridworld of arbitrary size and with arbitrary obstacles.
#Each state of the gridworld should have a reward. The transition matrix is defined
#as the probability of executing an action given a command to the robot.

import numpy as np

class GridWorld:

    def __init__(self, tot_row=5, tot_col=5):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        #The world is a matrix of size row x col x 2
        #The first layer contains the obstacles
        #The second layer contains the rewards
        #self.world_matrix = np.zeros((tot_row, tot_col, 2))
        #self.transition_array = np.ones(self.action_space_size) / self.action_space_size
        self.state_matrix = np.zeros((tot_row, tot_col))
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]
        #self.transition_array = transition_array
        
        
    def setStateMatrix(self, state_matrix):
        '''Set the obstacles in the world.

        The input to the function is a matrix with the
        same size of the world 
        -1 for states which are not walkable.
        +1 for terminal states
         0 for all the walkable states (non terminal)
        The following matrix represents the 4x3 world
        used in the series "dissecting reinforcement learning"
        [[0,  0,  0, +1]
         [0, -1,  0, +1]
         [0,  0,  0,  0]]
        '''
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix

    def setPosition(self, index_row=None, index_col=None):
        ''' Set the position of the robot in a specific state.

        '''
        if(index_row is None or index_col is None): self.position = [np.random.randint(self.world_row), np.random.randint(self.world_col)]
        else: self.position = [index_row, index_col]

    def render(self):
        ''' Print the current world in the terminal.

        O represents the robot position
        - respresent empty states.
        # represents obstacles
        * represents terminal states
        '''
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):
                if(self.position == [row, col]): row_string += u" \u25CB " # u" \u25CC "
                else:
                    if(self.state_matrix[row, col] == 0): row_string += ' - '
                    elif(self.state_matrix[row, col] == -1): row_string += ' # '
                    elif(self.state_matrix[row, col] == +1): row_string += ' * '
            row_string += '\n'
            graph += row_string 
        print (graph)
        
    def print_policy(self, policy_matrix):
        '''Print the policy using specific symbol.
        * terminal state
        ^ > v < up, right, down, left
        # obstacle
        '''
        counter = 0
        shape = policy_matrix.shape
        policy_string = ""
        for row in range(shape[0]):
            for col in range(shape[1]):
                if(policy_matrix[row,col] == -1): policy_string += " *  "            
                elif(policy_matrix[row,col] == 0): policy_string += " ^  "
                elif(policy_matrix[row,col] == 1): policy_string += " >  "
                elif(policy_matrix[row,col] == 2): policy_string += " v  "           
                elif(policy_matrix[row,col] == 3): policy_string += " <  "
                elif(np.isnan(policy_matrix[row,col])): policy_string += " #  "
                counter += 1
            policy_string += '\n'
        print(policy_string)

    def reset(self, exploring_starts=False):
        ''' Set the position of the robot in the bottom left corner.

        It returns the first observation
        '''
        if exploring_starts:
            while(True):
                row = np.random.randint(0, self.world_row)
                col = np.random.randint(0, self.world_col)
                if(self.state_matrix[row, col] == 0): break
            self.position = [row, col]
        else:
            self.position = [self.world_row-1, 0]
        #reward = self.reward_matrix[self.position[0], self.position[1]]
        return self.position

    def step(self, action):
        ''' One step in the world.

        [observation, reward, done = env.step(action)]
        The robot moves one step in the world based on the action given.
        The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        @return observation the position of the robot after the step
        @return reward the reward associated with the next state
        @return done True if the state is terminal  
        '''
        if(action >= self.action_space_size): `
            raise ValueError('The action is not included in the action space.')

        #Based on the current action and the probability derived
        #from the trasition model it chooses a new actio to perform
        action = np.random.choice(4, 1)
        #action = self.transition_model(action)

        #Generating a new position based on the current position and action
        if(action == 0): new_position = [self.position[0]-1, self.position[1]]   #UP
        elif(action == 1): new_position = [self.position[0], self.position[1]+1] #RIGHT
        elif(action == 2): new_position = [self.position[0]+1, self.position[1]] #DOWN
        elif(action == 3): new_position = [self.position[0], self.position[1]-1] #LEFT
        else: raise ValueError('The action is not included in the action space.')
        
        reward = -1
        #Check if the new position is a valid position
        #print(self.state_matrix)
        if (new_position[0]==0 and new_position[1]==1):
            reward = 30
            new_position[0]=3
            new_position[1]=1
        
        if (new_position[0]==0 and new_position[1]==3):
            reward = 10
            new_position[0]=2
            new_position[1]=3       
        
        if (new_position[0]>=0 and new_position[0]<self.world_row):
            if(new_position[1]>=0 and new_position[1]<self.world_col):
                self.position = new_position
                reward = 0
        #reward = self.reward_matrix[self.position[0], self.position[1]]
        #Done is True if the state is a terminal state
        #done = bool(self.state_matrix[self.position[0], self.position[1]])
        return self.position, reward

