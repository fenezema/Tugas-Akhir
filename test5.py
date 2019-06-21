# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 01:04:45 2019

@author: fenezema
"""

class Test():
    def __init__(self):
        self.atrb = None
        
    def testt(self,toggle=False,size=0):
        if toggle==True:
            print(size)
        else:
            print(toggle)
            
hehe = Test()
hehe.testt(toggle=True,size=32)