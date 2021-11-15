# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:22:09 2021

@author: Allahyar
"""

import numpy as np
import scipy as sp


class Nested_Sampling():
    def __init__(self,L,prior,dimension,n_live=10,tol=0.00000000000000001):
        #define likelihood function, prior, minimum likelihood, 
        #define dimension of parameter space, live point number tol and compression factor
        #live points
        self.n_live=n_live
        #compression factor
        self.t=np.exp(-1/(self.n_live))
        
        print('total live points chosen'+str(n_live))
        print('compression estimate chosen as'+str(self.t))
        self.X=1
        self.Z=0
        #likelihood function
        self.L=L
        #prior
        self.prior=prior
        #minimum likelihood
        #self.L*=L*
        #dimensionality
        self.dimension=dimension
        self.tol=tol
        
        
    def Vol_Contract(self):
        size= (self.n_live,self.dimension)
        live_points= np.random.uniform(0, 1, size) 
        print(live_points)
        startpoint=1
        while True:
            coord=0
            self.Low=self.L(live_points[0]) #initialise self.Low which is our L*. Lowest likelihood value
            for i in live_points: #find lowest L and set L* to it
                if self.Low>self.L(i):
                    self.Low=self.L(i)
                    lowest_array_position= coord
                coord=coord+1
            print("lowest coordinate is"+str(lowest_array_position))
            print("L* is"+str(self.Low))
            while True: #finding point at higher than L* to repopulate with
                new_point = np.random.uniform(0, 1, self.dimension)
                if self.L(new_point)>self.Low:
                    live_points[lowest_array_position]=new_point
                    break
            self.Z_0=self.Z
            self.Z=self.Z+(1-self.t)*self.X*self.Low #iterating the volume of new contour onto Z
            self.X=self.t*self.X
            print(self.Z)
            if abs(self.Z-self.Z_0)<self.tol: #stopping criteria of integral converging to 9 dp
                print(self.Z)
                break
                
        return self.Z
       # for j in live_points:
            #go through all the live points j, and repopulate...
            #...the ones with L<L*
        #    if self.L(j)<self.L*:
                
                    
            
           
            
#            
#
#    def f(self,h,*theta):
#        self.h=h
#        return self.h
    

#Below choose your desired likelihood function
def Likelihood(theta):
    return np.exp(-((theta[0]-0.5)**2+(theta[1]-0.5)**2)/(0.02))

testrun = Nested_Sampling(L=Likelihood,prior=1,n_live=10000,dimension=2)
integral= testrun.Vol_Contract()