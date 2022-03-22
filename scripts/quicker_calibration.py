# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:32:37 2021

@author: druth
"""

import matplotlib.pyplot as plt
import numpy as np

folder = r'W:\calibration\bulkstereo\stereo_calibration_20210602_A\40026941\\'
imname = 'y_400mm'

im = plt.imread(folder+imname+'.tiff')
im_shape = np.shape(im)


fig,ax = plt.subplots()
ax.imshow(im,cmap='gray')

#four_corners = plt.ginput(n=4,timeout=0)
four_corners = np.array(
      [[  66.000905  , 1018.33077248],
       [1381.56021852,  982.6594245 ],
       [1380.1333646 ,  176.48696014],
       [  51.7323658 ,  182.19437582]])

# bottom righ

#four_corners = np.concatenate([four_corners,np.atleast_2d(four_corners[0,:])])

ax.plot(four_corners[:,0],four_corners[:,1],'x',color='r')

l = -0.09
r = .12
b = -0.07
t = 0.06

n_x = int((r-l)*100) + 1
n_z = int((t-b)*100) + 1

points_XZ = np.array([[l,b],
                   [r,b],
                   [r,t],
                   [l,t],])

# # bottom
# px = np.linspace(four_corners[0,:],four_corners[1,:],n_x)
# ax.plot(px[:,0],px[:,1],'o',color='b',alpha=0.5)

# # top
# px = np.linspace(four_corners[2,:],four_corners[3,:],n_x)
# ax.plot(px[:,0],px[:,1],'o',color='b',alpha=0.5)

# # left
# px = np.linspace(four_corners[3,:],four_corners[0,:],n_z)
# ax.plot(px[:,0],px[:,1],'o',color='b',alpha=0.5)

# # right
# px = np.linspace(four_corners[1,:],four_corners[2,:],n_z)
# ax.plot(px[:,0],px[:,1],'o',color='b',alpha=0.5)

# predict x,y as functions of (X,Z)
import scipy.interpolate
X_new = np.arange(l,r+0.001,0.01,)
Z_new = np.ones_like(X_new) * 0
pred_x = scipy.interpolate.griddata(points_XZ,four_corners[:,0],np.array([X_new,Z_new]).T)
pred_z = scipy.interpolate.griddata(points_XZ,four_corners[:,1],np.array([X_new,Z_new]).T)
ax.plot(pred_x,pred_z,'x',color='orange')
stophere