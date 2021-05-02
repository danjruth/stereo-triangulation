# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:03:49 2021

@author: druth
"""

import numpy as np
from stereo.camera.camera import coefs_to_points
import scipy.optimize

DEFAULT_SPATIAL_LIMS = dict(x=np.array([-np.inf,np.inf]),
                            y=np.array([-np.inf,np.inf]),
                            z=np.array([-np.inf,np.inf]))
class StereoSystem:
    '''
    A class to combine two CameraCalibrations into a stereo vision system.
    
    Parameters
    ----------
    calib_A,calib_B : CameraCalibration
        The two camera calibrations.
        
    lims : dict, optional
        The known spatial limits of the domain, given as lims[direction] = 
        [lower_limit,higher_limit]. Defaults to an infinite domain as given in
        DEFAULT_SPATIAL_LIMS, with entries overwritten with whichever are given
        in lims.
    '''
    
    def __init__(self,calib_A,calib_B,lims=DEFAULT_SPATIAL_LIMS):        
        self.calibs = [calib_A,calib_B]
        
        # update the spatial limits
        _lims = DEFAULT_SPATIAL_LIMS.copy()
        for key in lims:
            _lims[key] = lims[key]
        self.lims=_lims
        
    def find_shortest_connection(self,xy_A,xy_B):        
        '''
        Given pixel coords for the two cameras, find the shortest line segment
        connection the two rays.
        '''
        coefs_A = self.calibs[0].linear_ray_coefs(xy_A[0],xy_A[1])
        coefs_B = self.calibs[1].linear_ray_coefs(xy_B[0],xy_B[1])
        return shortest_connection_two_rays(coefs_A,coefs_B)
    
    def __call__(self,xy_A,xy_B):
        '''
        Given pixel coords for the two cameras, find the probable 3d location
        and the error distance between the two rays.
        '''
        connection = self.find_shortest_connection(xy_A,xy_B)
        return connection_midpoint(connection), connection_dist(connection)
    
def shortest_connection_two_rays(coefs_A,coefs_B):
    '''
    Get the closest point between two lines, each defined as a function of Y.
    '''
    
    # name the coefficients from the arrays
    [[ax1,bx1],[_,_],[az1,bz1]] = coefs_A
    [[ax2,bx2],[_,_],[az2,bz2]] = coefs_B
    
    # see solving_shortest_connection_two_lines.nb in mathematica
    
    Y_1 = -1*((ax1*bx1 - ax2*bx1 - ax2*az1*az2*bx1 + ax1*az2**2*bx1 - ax1*bx2 + 
             ax2*bx2 + ax2*az1*az2*bx2 - ax1*az2**2*bx2 + az1*bz1 + 
             ax2**2*az1*bz1 - az2*bz1 - ax1*ax2*az2*bz1 - az1*bz2 - 
             ax2**2*az1*bz2 + az2*bz2 + ax1*ax2*az2*bz2)/(ax1**2 - 2*ax1*ax2 + 
             ax2**2 + az1**2 + ax2**2*az1**2 - 2*az1*az2 - 2*ax1*ax2*az1*az2 + 
             az2**2 + (ax1**2)*(az2**2)))
    
    Y_2 = -1*((ax1*bx1 - ax2*bx1 - ax2*az1**2*bx1 + ax1*az1*az2*bx1 - ax1*bx2 + 
               ax2*bx2 + ax2*az1**2*bx2 - ax1*az1*az2*bx2 + az1*bz1 + 
               ax1*ax2*az1*bz1 - az2*bz1 - ax1**2*az2*bz1 - az1*bz2 - 
               ax1*ax2*az1*bz2 + az2*bz2 + ax1**2*az2*bz2)/(ax1**2 - 2*ax1*ax2 + 
               ax2**2 + az1**2 + ax2**2*az1**2 - 2*az1*az2 - 2*ax1*ax2*az1*az2 + 
               az2**2 + (ax1**2)*(az2**2)))
    
    # return the points along the two rays
    return np.array([coefs_to_points(coefs_A,Y_1),
                     coefs_to_points(coefs_B,Y_2)])
    
def connection_midpoint(connection):
    return np.mean(connection,axis=0)
    
def connection_dist(connection):
    return np.linalg.norm(np.diff(connection,axis=0))

def dist_to_line(x0,coefs):
    '''
    Return the distance from a point to a line
    '''
    
    x1 = coefs_to_points(coefs,0)
    x2 = coefs_to_points(coefs,1)
    
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    d = np.linalg.norm(np.cross(x0-x1,x0-x2))/np.linalg.norm(x2-x1)
    return d

def minimize_dist_n_lines(linear_ray_coefs):
    '''
    Find the point closest to n lines given by the list linear_ray_coefs
    '''
    
    # a function which returns the error given the point location
    def err(x0):
        return np.sqrt(np.sum([dist_to_line(x0,coef)**2 for coef in linear_ray_coefs]))
    
    # minimize the error to get the closest point
    res = scipy.optimize.minimize(err,np.array([0,0,0]))
    
    return res.x,err(res.x)



def get_image_XYZ_coords(x_px,y_px,calib,XYZ_ref,xy_ref):
    '''
    Get the (X,Y,Z) coordinates of each pixel in an image, when the image is
    placed at a 3d reference location and oriented normal to the axis of the 
    camera which captured it. Can be used to with matplotlib's plot_surface to
    show a correctly-oriented image in 3d.
    
    Parameters
    -----------
    x_px,y_px : array-like
        The x and y pixel locations of each pixel in the image
        
    calib : stereo.CameraCalibration
        The camera calibration instance for the camera that captured the image
        
    XYZ_ref : array-like
        The (X,Y,Z) reference location of the pixel at xy_ref
        
    xy_ref : array-like
        The pixel (x,y) location at which the physical location is XYZ_ref
        
    Returns
    -----------
    xyz_im : np.array
        A numpy array of shape (len(y_px),len(x_px),3), where xyz_im[yi,xi,:] 
        is the (X,Y,Z) location of the pixel at (x_px[xi], y_px[yi])
    
    '''
        
    # get the ray coefficients going through the reference point of the image
    ray_coefs = calib.linear_ray_coefs(xy_ref[0],xy_ref[1])
    [[ax_ref,bx_ref,],[_,_],[az_ref,bz_ref]] = ray_coefs
    
    # get the pixel coordinates of each pixel in the image
    X_px,Y_px = np.meshgrid(x_px,y_px)
    xyz_im = np.zeros((len(y_px),len(x_px),3))    
    xyz_im_corners = np.zeros((2,2,3))
    
    # find the XYZ location of the four corners of the image
    for xi,x in enumerate(x_px[[0,-1]]):
        for yi,y in enumerate(y_px[[0,-1]]):
            
            xy_this_point = np.array([x,y])
            
            # calculate the ray going through this point
            ray_coefs_side = calib.linear_ray_coefs(xy_this_point[0],xy_this_point[1])
            [[axp,bxp,],[_,_],[azp,bzp]] = ray_coefs_side

            # find the Y value at which the ray intersects the reference plane
            Yp = (ax_ref*(XYZ_ref[0]-bxp) + az_ref*(XYZ_ref[2]-bzp)+XYZ_ref[1]) / (ax_ref*axp+az_ref*azp+1)
            
            # find the location of the intersection of this ray with the object's plane
            loc_p = coefs_to_points(ray_coefs_side,Yp)            
            xyz_im_corners[yi,xi,:] = loc_p
    
    # with the XYZ of the four corners computed, use linear interpolation to get all the pixel XYZs
    for i in [0,1,2]:
        xyz_im[...,i] = scipy.interpolate.interp2d(x_px[[0,-1]],y_px[[0,-1]],xyz_im_corners[...,i])(x_px,y_px).reshape(np.shape(X_px))
        
    return xyz_im

def find_epipolar_line_given_otherpx(calib_A,calib_B,xy_A,Y_vals):
    
    # corresponding XYZ values
    XYZ_vals = calib_A(xy_A[0],xy_A[1],Y_vals) # of line that corresponds to pixel in view A
    
    # get xy values in view B based on that inverse interpolator
    xy_b_vals = calib_B.inverse(XYZ_vals).T
    
    return xy_b_vals
