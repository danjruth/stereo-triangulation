# -*- coding: utf-8 -*-
"""
Created on Wed May 26 08:48:46 2021

@author: danjr
"""

import matplotlib.pyplot as plt
import scipy.interpolate

fig,axs = plt.subplots(2,2,figsize=(10,9))
x = 1000
y = 1000

for row_i, comp_i, phys_dir in zip([0,1],[0,2],['X','Z']):
        
    # known vs interpolated within the planes
    ax = axs[row_i,0]
    Y_vals = calib.object_points['Y'].unique()
    for yi,Y in enumerate(calib.object_points['Y'].unique()):
    
        cond = calib.object_points['Y']==Y
        Z_known = calib.object_points[phys_dir][:]    
        Z_pred = calib.calibration_planes(calib.image_points['x'][:].values,calib.image_points['y'][:].values)[yi,comp_i,:]
        
        err = Z_known - Z_pred
        points = np.array([calib.image_points['x'][:].values,calib.image_points['y'][:].values,calib.object_points['Y'][:].values]).T
        target = np.array([np.ones_like(Y_vals)*x,np.ones_like(Y_vals)*y,Y_vals]).T
        #err_line = scipy.interpolate.griddata([calib.image_points['x'][cond].values,calib.image_points['y'][cond].values,calib.object_points['Y'][cond].values],)
        res = scipy.interpolate.griddata(points.astype(float),err.values,target)
        
        ax.scatter((Z_known-Z_pred)*1000,np.ones_like(Z_known)*Y,label=str(int(Y*1000)),alpha=0.5)
        
    ax.set_xlabel('$'+phys_dir+'$ [m]')
    ax.set_ylabel('$'+phys_dir+'$ error [mm]')
    ax.legend(title='Plane $Y$ [mm]')
    ax.set_title('known vs interpolated in y planes')
    
    # known vs values from linear rays
    ax = axs[row_i,1]
    for yi,Y in enumerate(self.object_points['Y'].unique()):
        
        cond = self.object_points['Y']==Y
        Z_known = self.object_points[phys_dir][cond]
        Z_pred = []
        for i in np.arange(len(self.image_points[cond])):
            Z_pred.append(self.linear_ray_coefs.get_ray_segment(self.image_points['x'][cond].values[i],self.image_points['y'][cond].values[i],Y))

        Z_pred = np.array(Z_pred)[:,comp_i]
        ax.scatter(Z_known,(Z_known-Z_pred)*1000,label=str(int(Y*1000)),alpha=0.5)
    
    ax.set_xlabel('$'+phys_dir+'$ [m]')
    ax.set_ylabel('$'+phys_dir+'$ error [mm]')
    ax.legend(title='Plane $Y$ [mm]')
    ax.set_title('known vs values from linear rays')

if newfig:
    fig.tight_layout()