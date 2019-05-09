# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:47:00 2018

@author: nikki
"""

import pickle
with open('my_classifier.pkl', 'rb') as f:
    data = pickle.load(f)