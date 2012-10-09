'''
Created on 2012-9-27

@author: squall
'''
from numpy import *
def sigmoid(x):
    return mat(1.0/(1.0+exp(-x)))
def softMax(x):
    return mat(exp(x)/sum(exp(x)))
def logistic(x):
    return sigmoid(x)

if __name__=='__main__':
    a=mat(random.normal(size=25).reshape(5,5))
#    print sigmoid(a)
    b=mat(random.normal(size=25).reshape(5,5))
    c=a>b
    ff=lambda x :[int(a) for a in x]
    d=mat(map(ff,array(c)))
    a=random.normal(size=(10,20))
    print sum(a,0)