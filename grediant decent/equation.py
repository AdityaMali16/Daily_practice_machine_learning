import pandas as pd
import numpy as np
import math 


df=pd.read_csv('grediant decent/test_scores.csv')
print(df.head())

def gredecent(x,y):
    curr_coeff=curr_inter=0
    iteration=1000000
    learning_rate = 0.00021
    n=len(x)
    for i in range(0,iteration):
        yp=curr_coeff*x+curr_inter
        cost=(1/n)*np.sum((y-yp)**2)
        md=-(2/n)*np.sum(x*(y-yp))
        bd=-(2/n)*np.sum(y-yp)
        curr_coeff=curr_coeff-learning_rate*md
        curr_inter=curr_inter-learning_rate*bd
        print("m: {},       p: {},        cost: {},       iteration: {}".format(curr_coeff,curr_inter,cost,i))
    pass        





a=np.array(df['math'])
b=np.array(df['cs'])
gredecent(a,b)

