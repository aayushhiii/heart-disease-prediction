# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:05:04 2020

@author: Aashi
"""


import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':74, 'Gender':1, 'DM':1 , 'HT':1 , 'CRF':1 , 'Hypothrodism':0})

print(r.json())