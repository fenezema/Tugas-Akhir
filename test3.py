# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:37:10 2019

@author: fenezema
"""

from core import *
random_index = []            
n = 7
n_limit = int(30/100*n)
i=0
while i<n_limit:
    x = randint(0,n-1)
    if x not in random_index and x != 0:
        random_index.append(x)
        i+=1

print(random_index)