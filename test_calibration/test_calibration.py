# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:58:58 2021

@author: druth
"""

import numpy as np
import stereo.camera
import stereo.stereo_system.stereo_system as ssys

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

cam1 = stereo.camera.calib_from_dict(r'40026941.pkl')
cam2 = stereo.camera.calib_from_dict(r'40026942.pkl')

[cam.build_inverse_interpolator(cam.object_points.values.mean(axis=0)) for cam in [cam1,cam2]]

XYZ_use = cam1.object_points.values.mean(axis=0)
xy_A = cam1.inverse(XYZ_use)
xy_B = cam2.inverse(XYZ_use)
XYZ_use2 = cam2.object_points.values.mean(axis=0)
xy_A2 = cam1.inverse(XYZ_use2)
xy_B2 = cam2.inverse(XYZ_use2)

s = ssys.StereoSystem(cam1, cam2)

xA = [xy_A[0]+1,xy_A[0],xy_A2[0],40]
yA = [xy_A[1]-2,xy_A[1], xy_A2[1],90]

xB = [xy_B[0]+10,xy_B[0],xy_B2[0],10]
yB = [xy_B[1]-5,xy_B[1], xy_B2[1],30]

df_A = pd.DataFrame(dict(x=xA,y=yA,d_px=5))
df_B = pd.DataFrame(dict(x=xB,y=yB,d_px=5))

m = ssys.Matcher(df_A,df_B,s)
m.match()
stophere
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#cam1.draw_interpolant_lines(ax)
cam1.draw_calib_points(ax,color='r')
cam1.draw_bounding_rays(ax,color='r')
#cam2.draw_interpolant_lines(ax)
cam2.draw_calib_points(ax,color='b')
cam2.draw_bounding_rays(ax,color='b')