#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 01:20:59 2022

@author: sahibzadaallahyar
"""



# import timeit

# start = timeit.default_timer()

# #Your statements here

# stop = timeit.default_timer()

# print('Time: ', stop - start)  


import numpy as np
from anesthetic import MCMCSamples, NestedSamples
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
from scipy.stats import powerlaw
from scipy.special import logsumexp


#params
theta= [2,1]

m_true,c_true= theta
#error on response variable
sigma=0.1

#number of data points
N= 100
#number of clusters
K=3
#value of m in clustering/control variate method
cluster_nsub= 30


def f(x, theta):
      m, c = theta
      return m * (x)**3 + c
  

def fprime(x, theta):
      m, c = theta
      return 3*m*x**2 
  
    

def fprimeprime(x, theta):
      m, c = theta
      return 6*m*x
  
    
def f(x, theta):
      m, c = theta
      return m * x + c
  

def fprime(x, theta):
      m, c = theta
      return m 
  
    

def fprimeprime(x, theta):
      m, c = theta
      return 0



data_x = np.random.uniform(-1, 1, N)
data_y = f(data_x, [m_true, c_true]) + np.random.normal(0,sigma,N)




def loglikelihood(theta,coords):
      data_x= coords[:,0]
      data_y= coords[:,1]
      y = f(data_x,theta)
      # plt.scatter(data_x,data_y)
      logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y - y)**2/2/sigma**2).sum()
      # plt.figure(0)
      return logL



def loglikelihoodsimple(theta,n,Nsub = 10):
      i = np.random.choice(len(data_x),Nsub,replace=False)
      y = f(data_x[i],theta)
      # plt.scatter(data_x[i],data_y[i])
      logL = (-n/Nsub)*(np.log(2*np.pi*sigma**2)/2 + (data_y[i] - y)**2/2/sigma**2).sum()
      # plt.figure(1)
      return logL



cluster_x= np.linspace(-1,1,K)




class clustering():
    def __init__(self, f, f_prime, f_primeprime, x_data, y_data, cluster_covariates, theta_, Nsub_, loglikelihood_,sigma_,n):
        # you have to input np arrays for the data_x and data_y and cluster covariates 
        self.belongs_cluster=[]
        self.points = np.column_stack((x_data,y_data))
        self.theta=theta_
        self.Nsub= Nsub_
        cluster_y = f(cluster_covariates,theta)
        self.cluster_centres = np.column_stack((cluster_covariates,cluster_y))
        self.sigma = sigma_ 
        print('computation equivalent to Nsub='+str(Nsub_+7*len(cluster_covariates)))
        for m in self.points:
            distance=[]
            for k in self.cluster_centres:
                distance = np.concatenate((distance,np.sqrt(np.sum((k - m)**2))),axis=None)
            self.belongs_cluster.append(self.cluster_centres[np.argmin(distance)])
        zbar, n_k = np.unique(self.belongs_cluster,return_counts=True,axis=0)
        z_index=[]
        self.z_aligned=[]
        for mm in range(len(zbar)):
            self.z_aligned.append([])
        for j in range(len(zbar)):
            for i in range(len(self.points)):
                if (self.belongs_cluster[i] == zbar[j]).all():
                    self.z_aligned[j].append(self.points[i].tolist())        
        self.assigned_points= self.z_aligned.copy()
        self.delta_z = self.assigned_points.copy() 
        delta_l=[]
        sum1= 0
        sum2= 0
        sum3=0
        #create delta z k's
        for j in range(len(zbar)):
            self.delta_z[j] = (np.array(self.z_aligned[j]) - zbar[j])
            self.delta_zsum= np.sum(self.delta_z[j],axis=0)
            # print('delta z is'+str(self.delta_z[j]))
            sum1 += n_k[j]*loglikelihood_(theta=self.theta, coords=np.array([zbar[j]]))
            delta_l.append([])
            delta_x = self.delta_z[j][:,0]    
            delta_y = self.delta_z[j][:,1]
            delta_l[j] = [(1/(self.sigma**2))*(zbar[j][1]-f(zbar[j][0],self.theta)),(1/(self.sigma**2))*(zbar[j][1]-f(zbar[j][0],self.theta))*fprime(zbar[j][0], self.theta)]
            sum2 += np.dot(delta_l[j],self.delta_zsum)
            a = -((f_prime(zbar[j][0], self.theta))**2)+ (zbar[j][1]-f(zbar[j][0], self.theta))*f_primeprime(zbar[j][0], self.theta)
            b = f_prime(zbar[j][0],self.theta)
            c = -1
            sum3 += (1/(2*(self.sigma**2)))*(((delta_x**2)*a).sum() + (2*delta_y*delta_x*b).sum() + ((delta_y**2)*c).sum())
            print('the sums are')
            print(sum1,sum2,sum3)
            
            
            
            
        self.z_aligned2 = np.array([x for xs in self.z_aligned for x in xs])
        
        i2 = np.random.choice(len(self.z_aligned2),self.Nsub,replace=False)
        
        self.belongs_cluster2= np.array(self.belongs_cluster)[i2]
        self.points2 = self.z_aligned2[i2]
        sum_naive= (n/self.Nsub)*(loglikelihood_(self.theta, coords=self.points2))
        # sum_naive=0
        # print('points1')
        # print(self.z_aligned2[i2])
        # print('points2')
        # print(self.points2)
        self.logL = sum1+sum2+sum3+sum_naive
        
        zbar, n_k = np.unique(self.belongs_cluster2,return_counts=True,axis=0)
        self.z_aligned3=[]
        for mm in range(len(zbar)):
            self.z_aligned3.append([])
        for j in range(len(zbar)):
            for i in range(len(self.points2)):
                if (self.belongs_cluster2[i] == zbar[j]).all():
                    self.z_aligned3[j].append(self.points2[i].tolist())        
        self.delta_z2 = self.z_aligned3.copy() 
        delta_l=[]
        sum1= 0
        sum2= 0
        sum3=0
        
        for j in range(len(zbar)):
            self.delta_z2[j] = np.array(self.z_aligned3[j]) - zbar[j]
            print('delta z2 is')
            print(self.delta_z2[j])
            print(self.delta_z2[j][:,0])
            self.delta_zsum= np.sum(self.delta_z2[j],axis=0)
            # print('delta z is'+str(self.delta_z2[j]))
            sum1 += n_k[j]*loglikelihood_(theta=self.theta, coords=np.array([zbar[j]]))
            delta_l.append([])
            delta_x = self.delta_z2[j][:,0]    
            delta_y = self.delta_z2[j][:,1]
            delta_l[j] = [(1/(self.sigma**2))*(zbar[j][1]-f(zbar[j][0],self.theta)),(1/(self.sigma**2))*(zbar[j][1]-f(zbar[j][0],self.theta))*fprime(zbar[j][0], self.theta)]
            sum2 += np.dot(delta_l[j],self.delta_zsum)
            a = -((f_prime(zbar[j][0], self.theta))**2) + (zbar[j][1]-f(zbar[j][0], self.theta))*f_primeprime(zbar[j][0], self.theta)
            b = f_prime(zbar[j][0],self.theta)
            c = -1
            sum3 += (1/(2*(self.sigma**2)))*(((delta_x**2)*a).sum() + (2*delta_y*delta_x*b).sum() + ((delta_y**2)*c).sum())
            print('which is retarded',((delta_x**2)*a).sum(),(2*delta_y*delta_x*b).sum(),((delta_y**2)*c).sum())
            print('which is retarded',(delta_x**2),(2*delta_y*delta_x),((delta_y**2)))
            # print('which is retarded',(delta_x**2)*a,(2*delta_y*delta_x)*b,((delta_y**2))*c)
            print('which one is retarded',a,b,c)
            print('the m sums are')
            print(sum1,sum2,sum3,sum_naive,'SHOULD BE NEARRRRRR ',(n/self.Nsub)*(sum1+sum2+sum3))
        self.logL -= (n/self.Nsub)*(sum1+sum2+sum3)
        self.zbar =zbar
        print('ZBARRRRRRRRRRRRR',self.zbar)

    
    def coloured_points(self):
        return self.z_aligned3
    
    
    def zbarss(self):
        return self.zbar



    def ret(self):
        print('logL is'+str(self.logL))
        return self.logL
    
    
# clstr=clustering(f,fprime,fprimeprime,data_x,data_y,cluster_x,theta,10,loglikelihood,sigma,n=N)

# zk= clstr.ret()
colors = ['g','r','c','m','y','g','r','c','m','y']

clstr=clustering(f,fprime,fprimeprime,data_x,data_y,cluster_x,theta,cluster_nsub,loglikelihood,sigma,n=N)

zk= clstr.coloured_points()

zbars= clstr.zbarss()

for _ in range(len(zbars)):
    points_x= []
    points_y= []
    for o in range(len(zk[_])):
        points_x = np.concatenate((points_x,zk[_][o][0]),axis=None)
        points_y = np.concatenate((points_y,zk[_][o][1]),axis=None)
    plt.scatter(points_x, points_y, color= colors[_])
    

data_x1 = np.linspace(-1, 1,100000)
data_y1 = f(data_x1, [m_true, c_true])
plt.plot(data_x1,data_y1)

zbar_x=[]
zbar_y=[]
for _ in range(len(zbars)):
    points_x= []
    points_y= []

    zbar_x= np.concatenate((zbar_x,zbars[_][0]),axis=None)
    zbar_y= np.concatenate((zbar_y,zbars[_][1]),axis=None)
    
print('zBARRRRRRRRS',zbar_x,zbar_y)
plt.scatter(zbar_x,zbar_y,marker='x',color='black')
    

          
# for m in [1]:
#     plt.axvline(loglikelihoodsimple(theta,n=N,Nsub = N),color='k')
#     logL=np.ones(100)
#     for _ in range(100):
#         clstr=clustering(f,fprime,fprimeprime,data_x,data_y,cluster_x,theta,cluster_nsub,loglikelihood,sigma,n=N)
#         logL[_]= clstr.ret()
#     # logZ_all1=np.concatenate((logZ_all1,logZ_pl),axis=None)
#     logZ_pl=np.ones(100)
#     for _ in range(100):
#         logZ_pl[_]=loglikelihoodsimple(theta,n=N,Nsub= 10)
#     # logZ_all2=np.concatenate((logZ_all2,logZ_pl),axis=None)

# plt.hist(logL,color='red',alpha=0.1)
# error_sum1= np.std(logL)
# print('error on control variate subsampling',error_sum1,np.mean(logL))
# # plt.hist(logZ_pl, alpha=0.1,color='blue')
# error_sum2= np.std(logZ_pl)
# print('error naive sampling with equal sums',error_sum2)
# plt.subplots()




