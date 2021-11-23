 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

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
import pandas as pd
import matplotlib.pyplot as plt
#import operator
#import torch
import os
#import argparse
import tqdm #TODO: implement a progress bar for simulation runs 
import random 
#import sys
plt.style.use('seaborn-whitegrid')
#=========================================#
#Generate plots
#=========================================#


#AB

root = '/Users/kerner/Desktop/written_prelim/AB'
folder = os.listdir(root)
# for file in os.listdir(root):
#     path = os.path.join(root,file)

for num in range(6):
    num += 1
    path = '/Users/kerner/Desktop/written_prelim/AB/c{}.npy'.format(num)
    c1 = np.load(path)
    x = np.linspace(0,300,num=300)
    y = c1.mean(axis=0)
    plt.plot(x,y)
                

#Fu
root = '/Users/kerner/Desktop/written_prelim/Fu'
folder = os.listdir(root)
# for file in os.listdir(root):
#     path = os.path.join(root,file)

for num in range(6):
    num += 1
    path = '/Users/kerner/Desktop/written_prelim/Fu/c{}.npy'.format(num)
    c1 = np.load(path)
    x = np.linspace(0,300,num=300)
    y = c1.mean(axis=0)
    plt.plot(x,y)
                

#Random

root = '/Users/kerner/Desktop/written_prelim/Random'
folder = os.listdir(root)
# for file in os.listdir(root):
#     path = os.path.join(root,file)

for num in range(6):
    num += 1
    path = '/Users/kerner/Desktop/written_prelim/Random/c{}.npy'.format(num)
    c1 = np.load(path)
    x = np.linspace(0,300,num=300)
    y = c1.mean(axis=0)
    plt.plot(x,y)
    
 

policies = ['AB', 'Fu', 'Random']
conditions = {1: 'h = 0, B = .2',
              2: 'h = 0, B = .8',
              3: 'h = .25, B = .2',
              4: 'h = .25, B = .8',
              5: 'h = .5, B = .2',
              6: 'h = .5, B = .8'}


#=========================================#
#Compute table results 
#=========================================#
for num in range(6):
    plt.figure()
    
    num+=1
    for pol_name in policies:
        
        path = '/Users/kerner/Desktop/written_prelim/{}/c{}.npy'.format(pol_name, num)
        c = np.load(path)
        
        x = np.linspace(0,300,num=300)
        y = 10*c.mean(axis=0)
        plt.plot(x,y, label = '{}'.format(pol_name))
        plt.title(conditions[num])
        leg = plt.legend(loc='best')
        print('{}: Last 50 C{} is {}'.format(pol_name, num, y[:-50].mean()) )
        ybar = y.mean()
        print('{}: Full C{} is {}'.format(pol_name, num, ybar) )
        se = .1*sum(np.std(10*c, axis = 1))
        print('{}: Full SE{} is {}'.format(pol_name, num, se) )

    plt.show()

                
# for num in range(6):
    
    
#     num+=1
#     for pol_name in policies:
        
#         path = '/Users/kerner/Desktop/written_prelim/AB/lambdas/l{}.npy'.format( num)
#         c = np.load(path)
        
#         for i in range(3):
#             plt.figure()
#             x = np.linspace(0,300,num=300)
#             y = c[i,:].mean(axis=0)
#             plt.plot(x,y)
            
#         #leg = plt.legend(loc='best')
#     plt.show()



















