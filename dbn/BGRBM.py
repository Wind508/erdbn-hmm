#! coding=utf-8

'''
Created on 2012-9-27

@author: squall
'''

from tools import *
class BGRBM(object):
    '''
    the input is binary and the output is the guassian RBM
    and we assume the input data is a normal gauss
    '''


    def __init__(self,vis,hid,data,epochs,numcase,lr=0.001,show=True):
        '''
        Constructor
        '''
        self.vis=vis
        self.hid=hid
        self.data=data
        #initialize the weight 
        self.vis_hid=0.1*mat(random.normal(0,1,size=self.vis*self.hid).reshape(self.vis,self.hid))
        self.bias_vis=mat(zeros(self.vis))
        self.bias_hid=mat(zeros(self.hid))
        self.epochs=epochs
        self.numcase=numcase
        self.lr=lr
        self.batches=round(len(data)/self.numcase)
        
        self.momentum=0.5
        self.weightcost=0.0002
        self.show=show
    def popup(self,data):
        '''
        the different between BBRBM and BGRBM is :
        When we pop up the data, we do not need to compute the sigmoid of it 
        we just compute the mean of the new data
        '''
        eta=data*self.vis_hid+repeat(self.bias_hid.transpose(),self.numcase,1).transpose()
#        poshid=sigmoid(eta)
        return eta
    def popdown(self,data):
        eta=data*self.vis_hid.transpose()+repeat(self.bias_vis.transpose(),self.numcase,1).transpose()
        posvis=sigmoid(eta)
        return posvis
    def reconstruction_error(self,data):
        posvis=self.popdown(data)
        tmp=posvis>random.normal(0,1,self.vis)
        posvisstate=map(int,tmp)
        error=sum(posvisstate)-sum(data)
        return error
    def getstate(self,data,case,num):
        ff=lambda x:[int(a) for a in x]
        poshidstates=data>random.normal(0,1,(case,num))
        poshidstates=mat(map(ff,array(poshidstates)))
        return poshidstates
    def train(self):
        vishidinc  = zeros((self.vis,self.hid))
        hidbiaseinc = zeros(self.hid)
        visbiaseinc = zeros(self.vis)
        for i in range(self.epochs):
            
            errors=0.0
            for i in range(int(self.batches)):
                #positive phase
                tmp_data=self.data[i*self.numcase:(i+1)*self.numcase]
                #positive phase 
                poshidprobs=self.popup(tmp_data)
#                print poshidprobs
                posprobs=tmp_data.T*poshidprobs
                poshidact=sum(poshidprobs,0)
                posvisact=sum(tmp_data,0)
                # this is the different too
                poshidstates=poshidprobs+mat(random.normal(0,1,self.numcase*self.hid)).reshape(self.numcase,self.hid)
                #negative phase
                
#                print poshidprobs
                negdata=self.popdown(poshidstates)
                negdata=self.getstate(negdata,self.numcase,self.vis)
                negpos=self.popup(negdata)
                negprobs=negdata.T*negpos
                negvisact=sum(negdata,0)
                neghidact=sum(negpos,0)
                
                
                #use v0p0-v1p1 as the gradient
                vishidinc=self.momentum*vishidinc+self.lr*((posprobs-negprobs)/self.numcase-self.weightcost*self.vis_hid)
                hidbiaseinc=self.momentum*hidbiaseinc+(self.lr/self.numcase)*(poshidact-neghidact)
                visbiaseinc=self.momentum*visbiaseinc+(self.lr/self.numcase)*(posvisact-negvisact)
                self.vis_hid+=vishidinc
                self.bias_hid+=hidbiaseinc
                self.bias_vis+=visbiaseinc
                errors+=sum(pow(sum(tmp_data-negdata),2))
            print errors
if __name__=='__main__':
    a=mat(random.normal(size=100000)).reshape(1000,100)
    r=BGRBM(100,500,a,100,10)
    r.train()