import matplotlib.pyplot as plt
import numpy as np
import sys


number_stars=1000

x_ = np.random.uniform(-10,10,size = number_stars)
y_ = np.random.uniform(-10,10,size = number_stars)
z_ = np.random.uniform(10,20,size = number_stars)

def cart2sph(x, y, z):
   xy = np.sqrt(x**2 + y**2)
    
   x_2 = x**2
   y_2 = y**2
   z_2 = z**2

   r = np.sqrt(x_2 + y_2 + z_2) # r = sqrt(x² + y² + z²)

   theta = np.arctan2(y, x) 

   phi = np.arctan2(xy, z) 

   return theta, phi


g= np.column_stack((x_,y_,z_))

for x, y, z in g:
        
    r=(x**2 + y**2 + z**2)**0.5
    w = 1
    t = np.linspace(0,2*np.pi,1000)/w
    A = 0.1

    u_plus = x/z + A * x * (r*z + x**2 - y**2 + z**2) * np.cos((r+t)*w) / (w**2*(r+z)*z**2)
    v_plus = y/z + A * y * (r*z - x**2 + y**2 + z**2) * np.cos((r+t)*w) / (w**2*(r+z)*z**2)

    u_cross = x/z + A * y * (r*z - x**2 + y**2 + z**2) * np.cos((r+t)*w) / (w**2*(r+z)*z**2) 
    v_cross = y/z - A * x * (r*z + x**2 - y**2 + z**2) * np.cos((r+t)*w) / (w**2*(r+z)*z**2)
    
    # u_plus= x/z + A*x*(-w*(r+z)*(r**2+r*z-2*y**2)*np.cos((r+t)*w)+(r**2+2*r*z-2*y**2+z**2)*np.sin((r+t)*w)
    #                     -(r**2+2*r*z-2*y**2+z**2)*np.sin((t-z)*w))/(z**2*(r+z)**2*w**3)
    # v_plus= y/z - A*y*(w*(r+z)*(r**2-r*z-2*y**2-2*z**2)*np.cos((r+t)*w)-(r**2-2*r*z-2*y**2-3*z**2)*np.sin((r+t)*w)
    #                     +(r**2-2*r*z-2*y**2-3*z**2)*np.sin((t-z)*w))/(z**2*(r+z)**2*w**3)
    # z_plus = np.cos(t)*(1/np.cos(t))*z
    z_plus= np.ones_like(u_plus)
    #converting cartesian to spherical polar coords
    theta, phi = cart2sph(u_plus, v_plus, z_plus)
    #changing radius to 1 so effectively projecting everything onto a sphere
    R = 1 
    
    #coming back to Cartesian and flattening, or in other words just ignoring the Z coordinate
    proj_x= R*np.cos(theta)*np.sin(phi)
    
    proj_y=R*np.sin(theta)*np.sin(phi)
   
    if z>0:
        plt.plot(proj_x,proj_y, 'C0',lw=0.3)
        
        plt.gca().set_aspect('equal')
    
   
    
    



for n, m, z in g:
    r = (x**2 + y**2 + z**2)**0.5
    w = 1
    t = np.linspace(0,2*np.pi,1000)/w
    A = 0.1
    
    x = m
    y= -n
    
    u_cross= n/z + A*x*(-w*(r+z)*(r**2+r*z-2*y**2)*np.cos((r+t)*w)+(r**2+2*r*z-2*y**2+z**2)*np.sin((r+t)*w)-(r**2+2*r*z-2*y**2+z**2)*np.sin((t-z)*w))/(z**2*(r+z)**2*w**3)
    v_cross= m/z - A*y*(w*(r+z)*(r**2-r*z-2*y**2-2*z**2)*np.cos((r+t)*w)-(r**2-2*r*z-2*y**2-3*z**2)*np.sin((r+t)*w)+(r**2-2*r*z-2*y**2-3*z**2)*np.sin((t-z)*w))/(z**2*(r+z)**2*w**3)
    # z_cross = np.cos(t)*(1/np.cos(t))*z
    z_cross= np.ones_like(u_plus)
    
    u_cross = n/z + A * m * (r*z - n**2 + m**2 + z**2) * np.cos((r+t)*w) / (w**2*(r+z)*z**2) 
    v_cross = m/z - A * n * (r*z + n**2 - m**2 + z**2) * np.cos((r+t)*w) / (w**2*(r+z)*z**2)
    
    theta = np.arctan(v_cross/u_cross)
    phi = np.arctan(np.sqrt(u_cross**2+v_cross**2)/z_cross)
    
    theta, phi = cart2sph(u_cross, v_cross, z_cross)
    R = 1 
    proj_x= R*np.cos(theta)*np.sin(phi)
    proj_y= R*np.sin(theta)*np.sin(phi)
        
    if z>0:
        plt.plot(proj_x,proj_y, 'C1',lw=0.3) 
        plt.gca().set_aspect('equal')