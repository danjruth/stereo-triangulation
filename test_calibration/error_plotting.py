# -*- coding: utf-8 -*-
"""
Created on Wed May 26 08:48:46 2021

@author: danjr
"""

import matplotlib.pyplot as plt
import scipy.interpolate
from wwudel.run.bulkbubbles.intermediate import load_stereo_system
import numpy as np

calib_folder = r'E:\210525\\'
calib_name = r'stereo_calibration_D'

# rise velocity
#calib_folder = r'E:\200722\\'
#calib_name = r'calibration_H'

ss = load_stereo_system(calib_folder,calib_name)
calib = ss.calibs[1]

fig,axs = plt.subplots(2,2,figsize=(10,9),sharex=False,sharey=False)

for _ in range(50):
    
    x = np.random.uniform(0,1440)
    y = np.random.uniform(0,1080)
    #x = 200
    #y = 1500

    for row_i, comp_i, phys_dir in zip([0,1],[0,2],['X','Z']):
            
        # known vs interpolated within the planes
        ax = axs[0,row_i]
        Y_vals = calib.object_points['Y'].unique()
        Z_known = calib.object_points[phys_dir].values # clicked values
        Z_pred = []
        for yi,Y in enumerate(calib.object_points['Y'].unique()):
            cond = calib.object_points['Y']==Y
            Z_pred.append(calib.calibration_planes(calib.image_points['x'][cond].values,calib.image_points['y'][cond].values)[yi,comp_i,:]) # interpolated with calibration planes
            #Z_pred.append([calib(xa,ya,Y=np.array([Y,Y]))[0,comp_i] for xa,ya in zip(calib.image_points['x'][cond].values,calib.image_points['y'][cond].values)])
        Z_pred = np.concatenate(Z_pred)
                       
            #Z_pred = calib.calibration_planes(calib.image_points['x'][:].values,calib.image_points['y'][:].values)[yi,comp_i,:]
            
        err = Z_known - Z_pred
        points = np.array([calib.image_points['x'][:].values,calib.image_points['y'][:].values,calib.object_points['Y'][:].values]).T
        target = np.array([np.ones_like(Y_vals)*x,np.ones_like(Y_vals)*y,Y_vals]).T
        #err_line = scipy.interpolate.griddata([calib.image_points['x'][cond].values,calib.image_points['y'][cond].values,calib.object_points['Y'][cond].values],)
        res = scipy.interpolate.griddata(points.astype(float),err,target)
        
        ax.plot(res*1000,target[:,-1],'-o',label='('+str(int(x))+', '+str(int(y))+')',alpha=0.2)
        ax.set_xlabel('$'+str(phys_dir)+'$ error [mm]')
        ax.set_ylabel('$Y$ [m]')
        ax.set_title('known vs interpolated within plane')
        
        
        # known vs predicted
        ax = axs[1,row_i]
        Y_vals = calib.object_points['Y'].unique()
        Z_known = calib.object_points[phys_dir].values # clicked values
        Z_pred = []
        for yi,Y in enumerate(calib.object_points['Y'].unique()):
            cond = calib.object_points['Y']==Y
            #Z_pred.append(calib.calibration_planes(calib.image_points['x'][cond].values,calib.image_points['y'][cond].values)[yi,comp_i,:]) # interpolated with calibration planes
            Z_pred.append([calib(xa,ya,Y=np.array([Y,Y]))[0,comp_i] for xa,ya in zip(calib.image_points['x'][cond].values,calib.image_points['y'][cond].values)])
        Z_pred = np.concatenate(Z_pred)
                       
            #Z_pred = calib.calibration_planes(calib.image_points['x'][:].values,calib.image_points['y'][:].values)[yi,comp_i,:]
            
        err = Z_known - Z_pred
        points = np.array([calib.image_points['x'][:].values,calib.image_points['y'][:].values,calib.object_points['Y'][:].values]).T
        target = np.array([np.ones_like(Y_vals)*x,np.ones_like(Y_vals)*y,Y_vals]).T
        #err_line = scipy.interpolate.griddata([calib.image_points['x'][cond].values,calib.image_points['y'][cond].values,calib.object_points['Y'][cond].values],)
        res = scipy.interpolate.griddata(points.astype(float),err,target)
        
        ax.plot(res*1000,target[:,-1],'-o',label='('+str(int(x))+', '+str(int(y))+')',alpha=0.2)
        ax.set_xlabel('$'+str(phys_dir)+'$ error [mm]')
        ax.set_ylabel('$Y$ [m]')
        ax.set_title('known vs predicted with CameraCalibration')
    
#axs[0,0].legend()
fig.tight_layout()

for ax in axs.flatten():
    ax.set_xlim(-0.3,0.3)
    
    #ax.scatter((Z_known-Z_pred)*1000,calib.object_points['Y'],label=str(int(Y*1000)),alpha=0.5)
        
#     ax.set_xlabel('$'+phys_dir+'$ [m]')
#     ax.set_ylabel('$'+phys_dir+'$ error [mm]')
#     ax.legend(title='Plane $Y$ [mm]')
#     ax.set_title('known vs interpolated in y planes')
    
#     # known vs values from linear rays
#     ax = axs[row_i,1]
#     for yi,Y in enumerate(self.object_points['Y'].unique()):
        
#         cond = self.object_points['Y']==Y
#         Z_known = self.object_points[phys_dir][cond]
#         Z_pred = []
#         for i in np.arange(len(self.image_points[cond])):
#             Z_pred.append(self.linear_ray_coefs.get_ray_segment(self.image_points['x'][cond].values[i],self.image_points['y'][cond].values[i],Y))

#         Z_pred = np.array(Z_pred)[:,comp_i]
#         ax.scatter(Z_known,(Z_known-Z_pred)*1000,label=str(int(Y*1000)),alpha=0.5)
    
#     ax.set_xlabel('$'+phys_dir+'$ [m]')
#     ax.set_ylabel('$'+phys_dir+'$ error [mm]')
#     ax.legend(title='Plane $Y$ [mm]')
#     ax.set_title('known vs values from linear rays')

# if newfig:
#     fig.tight_layout()