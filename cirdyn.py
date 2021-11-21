#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 15 17:21:25 2021

@author: Conor M. Artman
"""

#=========================================#
#Circulant Dynamics Example
#=========================================#

#Conor M. Artman
#November, 2021

#=========================================#
#Import libraries
#=========================================#

import numpy as np
#import os 
#import pandas as pd
import matplotlib.pyplot as plt
#import operator
#import torch
#import os
#import argparse
import tqdm #TODO: implement a progress bar for simulation runs 
import random 
#import sys

#print('Max size is {}'.format(sys.maxsize))

horizon = 1000
epochs = 1
N = 100
B = 20

#IDEA: shouldn't our objective be to min spread of reward earlier on, or at least balanced with, long term cum reward?

#TODO: how is budget incorporated right now for random and other polciies? What summaries to cinlude for MC sim?

#TODO: for time, maybe WIBQL, Fu, random, and myopic (sets a nice gradient of algs )

class CirculantDynamics:
    
    def __init__(self, trans_mat = None, 
                 policies = None, 
                 T = horizon, epsilon = .1, 
                 N = N,
                 seed = 0,
                 budget = B):
        
        random.seed(seed)
        if trans_mat == None:
            #Default circulant dynamics
            P1 = np.array([[.5,.5,0,0],[0,.5,.5,0],[0,0,.5,.5],[.5,0,0,.5]]).reshape(4,4)
            P0 = np.transpose(P1)
            self.P = {0:P0, 1:P1}
        self.epsilon = epsilon
        self.policies = policies
        self.T = int(T)
        self.N =int( N)
        self.R = {0: -1, 1: 0, 2: 0, 3: 1}
        self.policy_type = None
        self.B = int(budget )
        self.ab_count = np.zeros(self.N*4*2).reshape(self.N,4,2).astype(int)
        
        if self.policies is None:
            
            self.policy_type = 'all'
            

        if self.policy_type == 'all':
            #make a dict of all learning methods
            self.policies = {'wiql': self.wiql_update, 
                              'ab': self.ab_update, 
                              'fu': self.fu_update,
                              'random': self.random_update}
            
            self.avg_rewards = {'wiql': np.zeros(self.T), 
                              'ab': np.zeros(self.T), 
                              'fu': np.zeros(self.T),
                              'random': np.zeros(self.T)}
            self.regret = {'wiql': np.zeros(self.T), 
                              'ab': np.zeros(self.T), 
                              'fu': np.zeros(self.T),
                              'random': np.zeros(self.T)}
            
            self.lambdas = {'wiql': np.zeros(self.T), 
                              'ab': np.zeros(self.N, 4), 
                              'fu': np.random.rand(self.N, 4)}
            self.Q = {'wiql': np.zeros(self.T), 
                              'ab': np.zeros(self.T), 
                              'fu': np.random.rand(self.N, 4, 2, self.N)}

        if self.policies == 'random':
            
            self.policy_type = 'random'
            self.policies = {'random': self.random_update}
            self.avg_rewards = {'random': np.zeros(self.T)}
            self.Q = None
            self.lambdas = None
            
        if self.policies == 'fu':
            lamRange = np.linspace(-1.25, 1.25, num = 10)
            
            self.policies = { 'fu': self.fu_update}
            
            self.avg_rewards = {'fu': np.zeros(self.T)}
            self.regret = {'fu': np.zeros(self.T)}
            
            self.lambdas = {'fu': np.random.choice(lamRange, self.N*4).reshape(self.N,4)}
            
            self.Q = {'fu': np.random.rand(4, 2, 10)}
            
        if self.policies == 'ab':
            
            #TODO: add lambda time series to see the lambda's converge 
            
            self.policies = { 'ab': self.ab_update}
            
            self.avg_rewards = {'ab': np.zeros(self.T)}
            
            self.lambdas = {'ab': np.zeros(self.N*4).reshape(self.N,4)}
            
            #Initialize Q table as AB describe 
            init_r = np.random.choice([-1,0,0,1], size=self.N*4*4*2).reshape(self.N,4,4,2)
            self.Q = {'ab': init_r}
            
            
    def run(self):
        init_active = np.random.choice(range(self.N), size=self.B + 2)
        init = np.zeros(self.N)
        init[init_active] += 1 
        
        init_state = {'ab': init}
        #set initial arm state
        
        for t in range(self.T):
            
            if t == 0:
                self.arm_states = init_state
                next_states = init_state
            else:
                self.arm_states = next_states
            
            next_states = self.step(t, self.arm_states)
        
        #Do a summary

        #Return the summary 

        return self.avg_rewards
    
    def step(self, t, states):

        next_states = {}

        
        #Loop through applying each policy method 
            
        for pol_name, update_rule in zip(self.policies.keys(), self.policies.values()):   
            
            if pol_name == 'fu':
                if t > 0:
                    gamma = min(1, 2*((t)**(-.5)))
                else:
                    gamma = .5
                explore_outcome = bool(np.random.binomial(1, p = gamma))
                
                #TODO: adjust this for either policy to work 
                
                if explore_outcome:
                    actions = self.chooseActions(pol_name, states[pol_name])
                    np.where((actions==0)|(actions==1), actions^1, actions)
                else:
                    actions = self.chooseActions(pol_name, states[pol_name])
                    
            elif pol_name == 'ab':
                
                explore_outcome = bool(np.random.binomial(1, p = self.epsilon))
                actions = self.chooseActions(pol_name, states[pol_name], explore=explore_outcome)

            rewards = self.getRewards(states[pol_name])
            next_states[pol_name] = self.getNextStates(actions, states[pol_name])
            
            if pol_name == 'random':
                self.updateRewards(t, pol_name, rewards)
                
            #if pol_name == 'fu':
            else:
                
                self.updateRewards(t, pol_name, rewards)
                
                #Update Q and Lambda tables for this specific policy
                
                #Q
                update_rule(t, states[pol_name], next_states[pol_name], rewards, actions, updateQ = True)
                
                #Lambda
                update_rule(t, states[pol_name], next_states[pol_name], rewards, actions, updateLam = True, explore = explore_outcome)
                
    
        return next_states

    def chooseActions(self, pol_name, currentStates, explore = False):
        
        currentStates = currentStates.astype(int)
        #Choose active arms according to Whittle indices
        actions = np.zeros(self.N).astype(int)
        
        #Based on:
        #numbers = np.array([1, 3, 2, 4])    
        #n = 2
        #indices = (-numbers).argsort()[:n], returns 3 and 1
        
        if pol_name == 'random':
            
            active_choices = np.random.choice(range(self.N), size=self.B)
            actions = np.zeros(self.N)
            actions[active_choices] += 1 
            
        if pol_name == 'ab' and explore:
            
            active_choices = np.random.choice(range(self.N), size=self.B)
            actions = np.zeros(self.N)
            actions[active_choices] += 1 
            
        else:
            active_arms = self.lambdas[pol_name]
            activation_indices = (-active_arms).argsort()[:self.B]
            actions[activation_indices] += 1
        
        return actions

    def wiql_update(self):
        
        return
    
    def ab_update(self, t, currentStates, nextStates, rewards, currentActions,  updateQ = False, updateLam = False, explore = False, average = True):
        currentActions = currentActions.astype(int)
        currentStates = currentStates.astype(int)
        nextStates = nextStates.astype(int)
        rewards = rewards.astype(int)
        #learning_div = 500
        
        C = 0.01 #TODO: find what choice is best for this 
        
        
        lams = self.lambdas['ab'].astype(int)
        self._ab_count_update(currentStates, currentActions)
        #gamma = .99
        
        if updateQ:
            
            for arm in range(self.N):
                r = int(rewards[arm])
                action = currentActions[arm]
                for state in currentStates.astype(int):
                    
                    update_multiplier = C / np.ceil(self.ab_count[arm, state, action]/500)
                    
                    for s_i in range(4):
                        
                        if average:
                            f = self.Q['ab'][arm,:,:,:].mean()
                            #FIXME
                            self.Q['ab'][arm, s_i, state, action] += update_multiplier*(r - lams[arm, state] + self.Q['ab'][arm, s_i, nextStates[arm]].max() - f - self.Q['ab'][arm, s_i, state, currentActions[arm]])
                        
                        
                        #else:
                         #   self.Q['ab'][arm, s_i, state, action] += update_multiplier*(r - lams[arm, state] + gamma*self.Q['ab'][arm, s_i, nextStates[arm]].max() - self.Q['ab'][arm, s_i, state, currentActions[arm]])
        #FIXME 
        if t % (self.N) == 0 and updateLam:
            Cprime = .001
            #lambda_lb = -10
            #lambda_ub = 10
            update_multiplier = Cprime/ (1 + np.ceil(t*np.log(t)/ 500) )
            
            for arm in range(self.N):
                for state in currentStates.astype(int):
                        #lam = self.lambdas['ab'][arm, state]
                        Q1 = self.Q['ab'][arm, state, state, 1]
                        Q0 = self.Q['ab'][arm, state, state, 0]
                        
                        self.lambdas['ab'][arm, state] = update_multiplier*(Q1 - Q0)
                        
                        #self.lambdas['ab'][arm, state] = min(lambda_ub, lam[arm, state])
                        #self.lambdas['ab'][arm, state] = max(lambda_lb, lam[arm, state])
            
            
        return
    
    def _ab_count_update(self, currentStates, actions):
        
        for arm in range(self.N):
            
            for state in currentStates.astype(int):
                
                self.ab_count[arm, state, actions[arm]]+= 1
        
        
        return
    
    def fu_update(self, t, currentStates, nextStates, rewards, currentActions,  updateQ = False, updateLam = False, explore = False):
        nextStates = nextStates.astype(int)
        beta = .99
        
        if t ==0:
            alpha = 0.5**(-.5)
        else:
            alpha = t**(-.5)
            
        
        lams = self.lambdas['fu'].astype(int)
        
        
        if updateQ:
        #Update Q table
          

          for arm in range(self.N):
              r = int(rewards[arm])
              for state in currentStates.astype(int):
                  for action in currentActions.astype(int):
                      #print(action, state, r, arm, lams[arm,state])
                      self.Q['fu'][state, action, lams[arm, state]] += (1-alpha)*(self.Q['fu'][state, action, lams[arm, state]]) + alpha*(r - lams[arm, state] + beta * max(self.Q['fu'][nextStates[arm], :, lams[arm, state]]))

        if updateLam:
            
            
            lamRange = np.linspace(-1.25, 1.25, num = 10) #Space of lambdas considered 
            
            if explore:
                #update lambda table based on uniform random draw to satisfy sufficient exploration of state-action-lambda space
                #TODO: also permute actions
                
                #for arm in range(self.N):
                for state in currentStates.astype(int):
                    self.lambdas['fu'][:, state] = np.random.choice(lamRange, 1)
                
            else:
                
                #Find the argmin of Q functinos over the lambda space 
                #for arm in range(self.N):
                    
                for state in currentStates.astype(int):
                        
                    Q1 = self.Q['fu'][state, 1, :]
                    Q0 = self.Q['fu'][state, 0, :]
                    
                    
                    newLam = lamRange[np.argmin(abs(Q1 - Q0))]
                    
                    self.lambdas['fu'][:,state] = newLam

        return  
    
    def random_update(self, arm_states = None):

        #Randomly draw 1's and 0's according to a uniform multinomial(N) satisfying the budget 

        active_choices = np.random.choice(range(self.N), size=self.B)
        actions = np.zeros(self.N)
        
        actions[active_choices] += 1
        
        return actions 
        
    def getNextStates(self, actions, currentStates):

        nextStates = np.zeros(self.N)
        for arm in range(self.N):

            nextStates[arm] = np.random.choice([0,1,2,3], size = 1, p = self.P[actions[arm]][int(currentStates[arm]),:])
    
        return nextStates

    def getRewards(self, currentStates):

        rewards = np.zeros(self.N)
        for arm in range(self.N):
            rewards[arm] += self.R[currentStates[arm]]


        return rewards
    
    def updateRewards(self, t, pol_name, currentRewards):
        '''
        Updates the reward of one specific algorithm. 

        '''
        self.avg_rewards[pol_name][t] = np.mean(currentRewards)

        return
    


def analyze(sim):
    
    return 

if __name__=='__main__':
    sim = CirculantDynamics(policies = 'ab')
    # print(sim.P[0][3,2])
    #print(sim.P[1])
    # print(4**4)
    y=sim.run()
    x = np.linspace(0,1000,num=1000)
    plt.plot(x, y['ab'])
    
    
    
    
    
    
    
    
    
    
