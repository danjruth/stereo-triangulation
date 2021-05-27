# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:55:00 2021

@author: druth
"""

import numpy as np
import scipy.interpolate
import cv2
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class CameraCalibration:
    '''
    Calibration of a single camera with the method given in Machicoane et al
    2019.
    
    Parameters
    ----------
    
    object_points : DataFrame
        A DataFrame containing columns X,Y,Z that give the 3-D location of all
        the physical locations of the target points (for all the calibration
        planes). The index must corrsepond to that of image_points. The 
        calibration planes are taken as the n_Y unique values of 
        object_points['Y'].
        
    image_points : DataFrame
        A DataFrame containing the x and y pixel coordinates of imaged target.
        The first column is treated as x and the second column is treated as y,
        regardless of the column names. The index must correspond to that of
        object_points.
        
    im_shape : array-like
        The number of rows and columns of the camera image, as (n_y,n_x).
        
    n_x_interpolate,n_y_interpolate : int
        The number of interpolating points to use between 0 and n_y and 0 and
        n_x.
    
    '''
    
    def __init__(self,object_points,image_points,im_shape,n_x_interpolate=10,n_y_interpolate=11):
        
        self.image_points = image_points.reset_index(drop=True)#[['x','y']].copy()
        #print('MULTIPLYING Y BY 10!!!!!!!!!!!!!!!!!')
        #object_points['Y'] = object_points['Y']*10
        self.object_points = object_points[['X','Y','Z']].copy().reset_index(drop=True)
        self.im_shape = im_shape
        self.Y_lims = np.array([self.object_points['Y'].min(),self.object_points['Y'].max()])
        
        # create the interpolating x and y values
        self.n_x_interpolate = n_x_interpolate
        self.n_y_interpolate = n_y_interpolate
        self.interpolant_x = np.linspace(0,im_shape[1],n_x_interpolate,endpoint=True)
        self.interpolant_y = np.linspace(0,im_shape[0],n_y_interpolate,endpoint=True)
        
        # calculate the (X,Z) location corresponding to the interpolant pixel values at each calibration plane
        I_XZ,I_x,I_y,Y_planes = get_px_to_transformed_locs_arr(self.object_points,
                                                                     self.image_points,
                                                                     self.interpolant_x,
                                                                     self.interpolant_y)
        self.calibration_planes = CalibrationPlanes(I_XZ,I_x,I_y,Y_planes)

        # get the linear coefficinets describing the rays through each image point
        linear_ray_coefs_arr, I_x, I_y = get_linear_ray_coeffs(self.calibration_planes,self.interpolant_x,self.interpolant_y)
        self.linear_ray_coefs = LinearRayCoefs(linear_ray_coefs_arr, I_x, I_y)
        
    def to_dict(self,fpath_save=None):
        '''Save the contents as a dict.
        '''
        
        d = {}
        d['object_points'] = self.object_points
        d['image_points'] = self.image_points
        d['im_shape'] = self.im_shape
        d['n_x_interpolate'] = self.n_x_interpolate
        d['n_y_interpolate'] = self.n_y_interpolate
        
        if fpath_save is not None:
            with open(fpath_save, 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        return d
        
    def __call__(self,x,y,Y=None):
        '''
        Get the endpoints of the linear fitted ray line segment.
        '''
        if Y is None:
            Y = self.Y_lims
        return self.linear_ray_coefs.get_ray_segment(x,y,Y)
    
    def set_axes_lims(self,ax,):
        '''
        Set the x and y limits for an axes corresponding to this calibration's
        image shape.
        '''
        ax.set_xlim(-0.5,self.im_shape[1]+0.5)
        ax.set_xlim(self.im_shape[0]+0.5,-0.5,)
        ax.set_aspect('equal')
    
    def draw_bounding_rays(self,ax,color='k',alpha=0.5,lw=3,ls='-',Y=None):
        if Y is None:
            Y = self.Y_lims
        # plot the fit straight line for these points
        linear_ray = self(0,0,Y=Y)
        ax.plot(linear_ray[:,0],linear_ray[:,1],linear_ray[:,2],ls,color=color,alpha=alpha,lw=lw)
        linear_ray = self(self.im_shape[1],0,Y=Y)
        ax.plot(linear_ray[:,0],linear_ray[:,1],linear_ray[:,2],ls,color=color,alpha=alpha,lw=lw)
        linear_ray = self(self.im_shape[1],self.im_shape[0],Y=Y)
        ax.plot(linear_ray[:,0],linear_ray[:,1],linear_ray[:,2],ls,color=color,alpha=alpha,lw=lw)
        linear_ray = self(0,self.im_shape[0],Y=Y)
        ax.plot(linear_ray[:,0],linear_ray[:,1],linear_ray[:,2],ls,color=color,alpha=alpha,lw=lw)
    
    def draw_interpolant_lines(self,ax,draw_calib_points=False,color='b',with_dots=True):

        for xi,x in enumerate(self.interpolant_x):
            for yi,y in enumerate(self.interpolant_y):                
                if with_dots:
                    for Yi,Y in enumerate(self.calibration_planes.Y_planes):
                        
                        # draw the transformed intersection with each plane
                        X,Z = self.calibration_planes.I_XZ[yi,xi,Yi,:]
                        ax.plot([X],[Y],[Z],'o',color=color,alpha=0.6)
                        
                    # call calc_ray_through_planes, which should align with these points
                    ray_points = self.calibration_planes(x,y)
                    ax.plot(ray_points[:,0],ray_points[:,1],ray_points[:,2],'.',color='orange',alpha=0.6)
                
                # plot the fit straight line for these points
                linear_ray = self(x,y)
                ax.plot(linear_ray[:,0],linear_ray[:,1],linear_ray[:,2],'-',color=color,alpha=0.6)
                
    def draw_calib_points(self,ax,marker='x',color='r',alpha=0.6):
        ax.plot(self.object_points['X'],self.object_points['Y'],self.object_points['Z'],marker,color=color,alpha=alpha)
            
    def build_inverse_interpolator(self,XYZ_center,bounds=np.array([-0.01,0.01]),err_thresh=1e-4):
        '''
        Get a method that uses interpolation to return the (x,y) pixel coords 
        of an object given its (X,Y,Z) physical coordinates. XYZ_center and 
        bounds define the (X,Y,Z) coordinates of the interpolating points used
        to build the interpolator.
        '''
        X = XYZ_center[0]+bounds
        Y = XYZ_center[1]+bounds
        Z = XYZ_center[2]+bounds
        self.inverse = build_inverse_interpolator(X,Y,Z,self,err_thresh=err_thresh)
        
    def plot_known_vs_predicted(self,axs=None):
        
        if axs is None:
            newfig = True
            fig,axs = plt.subplots(2,2,figsize=(10,9))
        else:
            newfig = False
        
        for row_i, comp_i, phys_dir in zip([0,1],[0,2],['X','Z']):
        
            # known vs interpolated within the planes
            ax = axs[row_i,0]
            for yi,Y in enumerate(self.object_points['Y'].unique()):
            
                cond = self.object_points['Y']==Y
                Z_known = self.object_points[phys_dir][cond]    
                Z_pred = self.calibration_planes(self.image_points['x'][cond].values,self.image_points['y'][cond].values)[yi,comp_i,:]
                ax.scatter(Z_known,(Z_known-Z_pred)*1000,label=str(int(Y*1000)),alpha=0.5)
                
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
            
    #def plot_err_vs_y(self,axs):
        
class CalibrationPlanes:
    '''
    Class for the interpolation of (X,Z) coords at the calibration planes. This
    is used as an intermediate step between calculating the (X,Z) location of 
    the interpolating points at each calibration Y plane and calculating the 
    linear ray coefficients at each interpolating point.
    
    Parameters
    ----------
    
    x,y : float
        The pixel coordinates of a point in the image
        
    interpd_XZ : np.ndarray
        Interpd values of (X,Z) at various Y planes for given interpolant pixel
        positions. Has shape (n_y,n_x,n_Y,2), where n_y and n_x are the number
        of interpolating rows and columns, and n_Y is the number of known
        calibration planes.
        
    I_x, I_y : np.ndarray
        2-d arrays of shape (n_y,n_x) given the pixel locations of the 
        interpolating points.
        
    Y_planes : np.ndarray
        1-d array of shape (n_Y) that gives the real-world Y coordinates of the 
        calibration planes.
    '''
    
    def __init__(self,I_XZ,I_x,I_y,Y_planes):
        
        self.I_XZ = I_XZ.copy()
        self.I_x = I_x.copy()
        self.I_y = I_y.copy()
        self.Y_planes = Y_planes.copy()
        
    def __call__(self,x,y):
        '''Calculate the (X,Y,Z) value of the ray's intersection with each
        plane via interpolation.
        '''
        ray_intersections = []
        for yi,Y in enumerate(self.Y_planes):
            # interpolate the (X,Z) location in this plane
            X = np.squeeze(scipy.interpolate.RectBivariateSpline(self.I_x[0,:],self.I_y[:,0],self.I_XZ[:,:,yi,0].T,kx=1,ky=1)(x,y,grid=False))
            Z = np.squeeze(scipy.interpolate.RectBivariateSpline(self.I_x[0,:],self.I_y[:,0],self.I_XZ[:,:,yi,1].T,kx=1,ky=1)(x,y,grid=False))
            #X = np.squeeze(np.diag(scipy.interpolate.interp2d(self.I_x,self.I_y,self.I_XZ[:,:,yi,0],kind='linear')(x,y)))
            #Z = np.squeeze(np.diag(scipy.interpolate.interp2d(self.I_x,self.I_y,self.I_XZ[:,:,yi,1],kind='linear')(x,y)))
            ray_intersections.append(np.array([X,Y*np.ones_like(X),Z]))
        return np.array(ray_intersections)
        
class LinearRayCoefs:
    '''
    Class for interpolating the coefficients describing the linear rays going
    through any point in the image. Stores the values with which the 
    interpolation is done, and contains the __call__ method to compute the 
    values at a given point.
    '''
    
    def __init__(self,interp_coefs,I_x,I_y):
        self.interp_coefs = interp_coefs.copy()
        self.I_x = I_x.copy()
        self.I_y = I_y.copy()
        
    def _get_coefs(self,x,y,viz=False,spline_order=3):
                
        # get the x coefs
        aX = scipy.interpolate.RectBivariateSpline(self.I_x[0,:],self.I_y[:,0],self.interp_coefs[:,:,0,0].T,kx=spline_order,ky=spline_order)(x,y,grid=False)
        bX = scipy.interpolate.RectBivariateSpline(self.I_x[0,:],self.I_y[:,0],self.interp_coefs[:,:,0,1].T,kx=spline_order,ky=spline_order)(x,y,grid=False)
        
        # get the z coefs
        aZ = scipy.interpolate.RectBivariateSpline(self.I_x[0,:],self.I_y[:,0],self.interp_coefs[:,:,2,0].T,kx=spline_order,ky=spline_order)(x,y,grid=False)
        bZ = scipy.interpolate.RectBivariateSpline(self.I_x[0,:],self.I_y[:,0],self.interp_coefs[:,:,2,1].T,kx=spline_order,ky=spline_order)(x,y,grid=False)
        
        coefs = np.zeros((3,2))
        coefs[0,:] = [aX,bX]
        coefs[1,:] = [1,0]
        coefs[2,:] = [aZ,bZ]
        
        return coefs
        
    def __call__(self,x,y,viz=False):
        
        if np.isscalar(x):
            return self._get_coefs(x,y,viz=viz)
        
        elif x.ndim==1:
            n_rows = len(x)
            all_coefs = np.zeros((n_rows,3,2))
            for i in np.arange(n_rows):
                all_coefs[i,...] = self._get_coefs(x[i],y[i],viz=False)
            return all_coefs
            
        else:
            print('x and y must be scalars or 1-d arrays')
            return
            
    def get_ray_segment(self,x,y,Y_line):
        coefs = self(x,y)
        return coefs_to_points(coefs,Y_line)
    
def calc_dx(xy_pix,point,calib,d_px=1,axes=None):
    '''
    Calculate the effective pixel size, given an object's 3d location and the
    pixel location.
    
    This is done by calculating the adjacent pixels' rays' intersections with
    the object's plane, then computing the real-world distance between these
    points.
    
    Parameters
    ----------
    
    xy_pix : tuple
        The (x,y) image cooridnates of the object.
        
    point : tuple
        The (X,Y,Z) coordinates of the object in 3D space.
        
    calib : CameraCalibration
        The calibration object for the camera.
        
    d_px : float, optional
        The small pixel displacement used to calculate the pixel size
        
    axes : matplotlib.Axes or None, optional
        The axes on which to illustrate the calculation.
        
    Returns
    ----------
    
    dx : float
        The mean of the pixel sizes in the camera's x and y directions.    
    '''
    
    # get the ray coefficients for this point
    ray_coefs = calib.linear_ray_coefs(xy_pix[0],xy_pix[1])
    [[ax,bx,],[_,_],[az,bz]] = ray_coefs
    [X,Y,Z] = point

    # do it once vertically, once horizontally
    dx_directional = []
    for axis in [0,1]:
        
        side_points = []
        for d in [-d_px,d_px]:
            
            # calculate the pixel coordinates of this "test point"
            xy_pix_side = np.array(xy_pix).astype(float)
            xy_pix_side[axis] = xy_pix_side[axis]+d
            
            # calculate the ray going through this point
            ray_coefs_side = calib.linear_ray_coefs(xy_pix_side[0],xy_pix_side[1])
            [[axp,bxp,],[_,_],[azp,bzp]] = ray_coefs_side

            # find the Y value at which the "right" ray intersects the object's plane
            Yp = (ax*(X-bxp) + az*(Z-bzp)+Y) / (ax*axp+az*azp+1)
            
            # find the location of the intersection of this ray with the object's plane
            loc_p = coefs_to_points(ray_coefs_side,Yp)
            side_points.append(loc_p)
            if axes is not None:
                axes.plot([loc_p[0]],[loc_p[1]],loc_p[[2]],'o',color='r')
                ray = calib(xy_pix_side[0],xy_pix_side[1])
                axes.plot(ray[:,0],ray[:,1],ray[:,2],'-',color='r')
            
        # get the displacement between the two points
        side_points_diff = np.diff(side_points,axis=0)
        dx_directional.append(np.linalg.norm(side_points_diff) / (2*d_px))
    return np.mean(dx_directional)
    
###############################################################################
### FUNCTIONS TO LOAD A SAVED CALIBRATION
###############################################################################
    
def calib_from_folder(folder,y_strs_mm,im_shape_yx,**kwargs):
    '''
    Initialize a CameraCalibration object given the folder its inputted data is
    store in.
    '''    
    dfs = pd.concat([pd.read_pickle(folder+'y_'+y+'mm.pkl') for y in y_strs_mm])
    calib = CameraCalibration(dfs[['X','Y','Z']],dfs[['x','y']],im_shape_yx,**kwargs)
    return calib
            
def calib_from_dict(d):
    '''
    Initialize a CameraCalibration object based on a dict which 
    '''
    
    # load the dict if a filepath was passed
    if isinstance(d, str):
        with open(d, 'rb') as handle:
            d = pickle.load(handle)
    
    calib = CameraCalibration(d['object_points'],
                              d['image_points'],
                              d['im_shape'],
                              n_x_interpolate=d['n_x_interpolate'],
                              n_y_interpolate=d['n_y_interpolate'])
    return calib
    
###############################################################################
### HELPER FUNCTIONS
###############################################################################

def apply_transformation_matrix(h,xi,yi):
    '''Apply a 3x3 transformation matrix to 2d points
    '''    
    denom = h[2,0]*xi + h[2,1]*yi + h[2,2]    
    x = (h[0,0]*xi + h[0,1]*yi + h[0,2]) / denom
    y = (h[1,0]*xi + h[1,1]*yi + h[1,2]) / denom    
    return np.moveaxis(np.array([x,y]),0,-1)

def get_px_to_transformed_locs_arr(object_points,image_points,interpolant_x,interpolant_y):
    '''
    At each calibration plane, calculate the (X,Z) location of each pixel (x,y)
    location used in the interpolation
    
    Parameters
    ----------
    
    object_points : DataFrame
        A DataFrame containing columns X,Y,Z that give the 3-D location of all
        the physical locations of the target points (for all the calibration
        planes). The index must corrsepond to that of image_points. The 
        calibration planes are taken as the n_Y unique values of 
        object_points['Y'].
        
    image_points : DataFrame
        A DataFrame containing the x and y pixel coordinates of imaged target.
        The first column is treated as x and the second column is treated as y,
        regardless of the column names. The index must correspond to that of
        object_points.
        
    interpolant_x,interpolant_y : np.ndarray
        1-D arrays giving the pixel x and y locations at which to calculate the
        (X,Z) location in each calibration plane. Their lengths set n_x and 
        n_y.
        
    Returns
    ----------
    
    trans_locs : np.ndarray
        An array of shape (n_y,n_x,n_Y,3) which gives the transformed locations
        of the interpolating pixel points (X,Y,Z) at each of the n_Y 
        calibration planes.
        
    I_x,I_y : np.ndarray
        The meshes of interpolant_x and interpolant_y.
        
    Y_planes : np.ndarray
        The Y locations of the n_Y calibration planes, corresponding to the
        third axis of trans_locs.
    '''
    
    Y_planes = np.sort(object_points['Y'].unique())

    # transformation for each plane
    plane_trans = {}
    #plane_tran_invs = {}
    for Y in Y_planes:
        
        # get the data for this plane
        use = object_points['Y']==Y    
        object_points_use = np.array(object_points.loc[use,['X','Z']].values)#[:4,:] ,dtype=np.float32
        image_points_use = np.array(image_points[use].values)#[:4,:] ,dtype=np.float32
        
        # get pixel -> real world transformations
        img_to_scene,_ = cv2.findHomography(image_points_use,object_points_use,)
        plane_trans[Y] = img_to_scene
                    
    # build the pixel-line interpolant
    I_x,I_y = np.meshgrid(interpolant_x,interpolant_y)
    
    # for ecah interpolant point, compute (X,Z) at each plane
    trans_locs = np.zeros((len(interpolant_y),len(interpolant_x),len(Y_planes),2))
    for yi,Y in enumerate(Y_planes):
        trans_locs[:,:,yi,:] = apply_transformation_matrix(plane_trans[Y],I_x,I_y)
        
    return trans_locs,I_x,I_y,Y_planes

def get_linear_ray_coeffs(ray_points_func,interpolant_x,interpolant_y):
    '''
    At various pixel locations given by interpolant_x and interpolant_y, find
    the coefficients describing the rays passing through that pixel as
    described by X = aX*Y+bX and Z = aZ*Y+bZ.
    
    Parameters
    ----------
    
    ray_points_func : callable
        The function returned by make_ray_points_func which takes pixel
        coordinates and returns all the intersections of the corresponding ray
        with the calibration planes.
        
    interpolant_x, interpolant_y : np.ndarray
        1-d arrays of the pixel x and y coordinates at which to calculate the
        coefficients. Their lengths define n_x and n_y, respectively.
        
    Returns
    ----------
    
    ray_coeffs : np.ndarray
        The coefficients for the linear fits to the ray. Has shape (n_y,n_x,3,
        2), where the ray passing through interpolating indices y_i,x_i is 
        given by:
            (X = ray_coeffs[yi,xi,0,0]*Y+ray_coeffs[yi,xi,0,1],
             Y = ray_coeffs[yi,xi,1,0]*Y+ray_coeffs[yi,xi,1,1] = 1*Y+0,
             Z = ray_coeffs[yi,xi,2,0]*Y+ray_coeffs[yi,xi,2,1])
            
    I_x, I_y : np.ndarray
        The meshes of the interpolating points.
    '''

    I_x,I_y = np.meshgrid(interpolant_x,interpolant_y)
    
    ray_coeffs = np.zeros((len(interpolant_y),len(interpolant_x),3,2)) # [y_i,x_i,real_axis,term]
    for xi,x in enumerate(interpolant_x):
        for yi,y in enumerate(interpolant_y):
            ray = ray_points_func(x,y)
            px = np.polyfit(ray[:,1],ray[:,0],1)
            pz = np.polyfit(ray[:,1],ray[:,2],1)
            ray_coeffs[yi,xi,0,:] = px
            ray_coeffs[yi,xi,1,:] = [1,0]
            ray_coeffs[yi,xi,2,:] = pz            
            
    return ray_coeffs, I_x, I_y

def coefs_to_points(coefs,Y):
    '''
    Get line segment coordinates given the ray coefficients and the Y values
    '''
    X = coefs[0,0]*Y+coefs[0,1]
    Z = coefs[2,0]*Y+coefs[2,1]
    return np.array([X,Y,Z]).T
            
def find_px_minimization(XYZ,calib):
    ''' find the (x,y) pixel location of the ray going through the point XYZ by
    minimization (takes a while)
    '''
    def err(xy):
        '''distance (in m) between the given XYZ and the approximated XYZ
        '''
        this_XYZ = calib(xy[0],xy[1],XYZ[1])
        return np.linalg.norm(XYZ-this_XYZ)
    res = scipy.optimize.minimize(err,(calib.im_shape[1]/2,calib.im_shape[0]/2))
    return res.x, err(res.x)
            
def build_inverse_interpolator(X,Y,Z,calib,err_thresh=1e-4):
    '''
    Get a function that returns the (x,y) pixel coordinates given (X,Y,Z) 
    object coordinates for a given camera.
    
    Parameters
    ----------
    
    X, Y, Z : 1-d np.ndarrays
        Arrays defining the grid of physical points used to build the 
        interpolator.
        
    calib : CameraCalibration
        The instance of a camera calibration on which the inverse interpolator
        is based.
        
    err_thresh : float
        The maximum error (in meters) between the given XYZ and the calculated 
        XYZ, given the (x,y) pixels found with find_px_minimization.
        
    Returns
    ----------
    
    interp : callable
        A function which returns pixel (x,y) coordinates corresponding to given
        XYZ locations, with the shape depending on the shape of XYZ
    '''
    
    print(np.shape(X))
    
    # get the xy points at each
    print('getting the xy points at each location...')
    XYZ_all = np.zeros((len(X),len(Y),len(Z),3))
    xy_px = np.zeros((len(X),len(Y),len(Z),2))
    err = np.zeros((len(X),len(Y),len(Z)))
    for xi,x in enumerate(X):
        for yi,y in enumerate(Y):
            for zi,z in enumerate(Z):
                XYZ = np.array([x,y,z])
                XYZ_all[xi,yi,zi,:] = XYZ
                xy_px[xi,yi,zi,:],err[xi,yi,zi] = find_px_minimization(XYZ,calib)
                
    # get rid of bad results
    mask = np.ones_like(err)
    # mask where the error is too big
    mask[err>err_thresh] = np.nan
    # mask where the returned (x,y) is outside the image shape
    mask[xy_px[...,0]<0] = np.nan
    mask[xy_px[...,1]<0] = np.nan
    mask[xy_px[...,0]>calib.im_shape[1]] = np.nan
    mask[xy_px[...,1]>calib.im_shape[0]] = np.nan
    # apply the mask
    xy_px[...,0] = xy_px[...,0]*mask
    xy_px[...,1] = xy_px[...,1]*mask
            
    # create inerpolators for the pixel x and y locations
    print('creating the interpolators...')
    [print(np.shape(a)) for a in (X,Y,Z)]
    interp_x = scipy.interpolate.RegularGridInterpolator((X,Y,Z),xy_px[...,0],bounds_error=False,fill_value=None,method='linear')
    interp_y = scipy.interpolate.RegularGridInterpolator((X,Y,Z),xy_px[...,1],bounds_error=False,fill_value=None,method='linear')
    
    
    # points = np.array([XYZ_all[...,i].flatten() for i in range(3)]).T
    # vals_x = xy_px[...,0].flatten()
    # vals_y = xy_px[...,1].flatten()
    # cond = (~np.isnan(vals_x)) * (~np.isnan(vals_y))
    # points = points[cond,:]
    # vals_x = vals_x[cond]
    # vals_y = vals_y[cond]
    # interp_x = scipy.interpolate.LinearNDInterpolator(points,vals_x,)
    # interp_y = scipy.interpolate.LinearNDInterpolator(points,vals_y,)
    
    # create a function for returning (x,y) given XYZ
    def interp(XYZ):
        return np.squeeze(np.array([interp_x(XYZ),interp_y(XYZ)]))
    
    # def interp(XYZ_find):
        
        
        
    #     print(np.shape(points))
    #     print(np.shape(vals_x))
    #     print(np.shape(XYZ_find))
    #     return np.squeeze(np.array([scipy.interpolate.griddata(points,vals_x,XYZ_find),scipy.interpolate.griddata(points,vals_y,XYZ_find)]))
    
    return interp

