# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:12:28 2019

@author: elina
"""

import pandas as pd
import numpy as np

import time

from sklearn.metrics import mean_squared_error

from collections import defaultdict
#from pathos.multiprocessing import ProcessingPool as Pool
import math
from sys import exit

class svd():
    def __init__(self, num_attr = 20,num_iters = 20, reg = 1e-3, lr = 3e-2):
        self.train=pd.read_csv("../train_ratings.csv")
        
        self.val=pd.read_csv("../val_ratings.csv")
        self.numu=self.train['userId'].astype(int).unique()#the length of users   
        self.numm=self.train['movieId'].astype(int).unique() #the length of movies 
        self.train['movieId']=self.train['movieId'].astype(int)
        self.train['userId']=self.train['userId'].astype(int)
        self.test=pd.read_csv("../val_ratings.csv")
        self.result=True
        self.mean=0
        self.userave=[]
        self.movieave=[]
        #self.computemean()

        self.iter=num_iters
        self.mean=self.train['rating'].mean() #overall mean
        self.val=pd.read_csv("../val_ratings.csv")
        self.vdic=pd.read_csv("moviehash2.csv", header=None) 
        ''' a hash of the movie tag to index, e.g. movie 7 is 0 on the index'''
        '''make it a dictionary for later access'''
        self.d = defaultdict(int)
        for i in range(self.vdic.shape[1]):
            self.d[int(self.vdic.iloc[0,i])]=int(self.vdic.iloc[1,i])
    
# =============================================================================
#         with open('result.json', 'r') as f:
#             self.usedic= json.load(f)
# =============================================================================
            
        self.prediction=[]
        

        

        
    def writepredict(self):
        '''write to submission '''
        self.test['rating']=self.rating
        self.test=self.test.drop(['userId', 'movieId'], axis=1)  
        outfile = open('submission_cf.csv', 'wb')
        self.test.to_csv('submission_cf.csv', index = False, header = True, sep = ',', encoding = 'utf-8')
        outfile.close()
        

        
    def pred2(self):
        '''predict value and/or write to submission '''
        stime=time.time()    
        m=self.test.shape[0]
        print(m)
        self.rating=np.zeros(m)
        for index, row in self.test.iterrows():
            i= int(row['userId'])
            k=int(row['movieId'])
            j=self.d[k]        
            if k in self.d.keys():
                j=self.d[k] 
                q=np.nan_to_num(np.dot(self.us[i, :],self.mo[:,j]))   
                bias=np.nan_to_num(self.userbias[i] + self.itembias[j])
                self.rating[index] = self.mean + bias +q
            else:
                self.rating[index]=self.mean 
            if np.isnan(self.rating[i]):
                self.rating[index]=self.mean  
            if (index % 10000 == 0):
                     print(str(i)+' time is '+str((time.time()-stime)/60)) 
        self.writepredict()
        #self.estimate()


    def train1(self,  K=11 , reg=0.02, w=0.002):
        
        '''train dataset with matrix factorization algorithm '''
        stime=time.time()         
# =============================================================================
#         mo = pd.DataFrame(np.random.rand(K, len(self.numm)),index=range(1,K+1), columns=self.numm)
#         us=  pd.DataFrame(np.random.rand(len(self.numu)+1, K),index=np.insert(self.numu,0,0), columns=range(1,K+1))
# =============================================================================
        mo=np.random.normal(scale = 1./20., size = (K, len(self.numm)))
        us=np.random.normal(scale = 1./20., size = (len(self.numu)+1, K))
        user_bias=np.random.normal(scale = 1./20, size = len(self.numu)+1)
        #user_bias=pd.read_csv('userbias.csv',  header=None).iloc[:,1]

        item_bias=np.random.normal(scale = 1./20, size = len(self.numm))
        '''initialize values'''
        

        print('start training!!!')
        for t in range(self.iter):
            print(str(t)+' allpredict '+str((time.time()-stime)/60)) 
            print('iteration: '+str(t)) 
            for index, row in self.train.iterrows():
                i= int(row['userId'])
                k=int(row['movieId'])
                j=self.d[k]
                r=row['rating']
                ub=user_bias[i]
                ib=item_bias[j]
                temp_mo=mo[:,j]
                q=np.nan_to_num(np.dot(us[i, :],temp_mo))                     
                prediction = self.mean +ib+ub+q         
                e = r - prediction
                we=w*e
                wreg=w*reg
                user_bias[i]+=np.nan_to_num(we - wreg * ub)
                item_bias[j]+=np.nan_to_num(we - wreg * ib) 
                temp1=np.nan_to_num(temp_mo*(we))
# =============================================================================
#                 temp1[temp1>20]=0.0001
#                 temp1[temp1<-20]=-0.0001                
# =============================================================================
                us[i, :]+=temp1
                temp2=np.nan_to_num(we * us[i, :])
# =============================================================================
#                 temp2[temp2>20]=0.0001
#                 temp2[temp2<-20]=-0.0001
# =============================================================================
                mo[:,j]+=temp2
                if (index % 50000 == 0):
                    print(str(index)+' time is '+str((time.time()-stime)/60))

            if t% 3==0:
                '''test prediction'''

                self.userbias=user_bias
                self.itembias=item_bias
                self.mo=mo
                self.us=us
                self.pred2()
                
        np.savetxt('userbias.csv', user_bias, delimiter=',')
        np.savetxt('moviebias.csv', item_bias, delimiter=',')
        np.savetxt('sgd_movie.csv', mo, delimiter=',')
        np.savetxt('sgd_user.csv', us, delimiter=',')

        self.userbias=user_bias
        self.itembias=item_bias
        self.mo=mo
        self.us=us
        self.pred2()
                
        
# =============================================================================
#         self.mo=mo
#         self.us=us
# =============================================================================



    def estimate(self):
        '''estimate the rmse'''
        print(math.sqrt(mean_squared_error(self.rating, self.test['rating'])))
        





def main():
    '''input is iteration number'''
    t=time.time()
    inp=input('iteration number: ')
    model = svd(num_iters=int(inp))
    model.train1()


    

main()

# =============================================================================
# 
# if __name__ == "__main__":
#     main()
# 
# =============================================================================
