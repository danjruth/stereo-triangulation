# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:03:49 2021

@author: druth
"""

import numpy as np
from stereo.camera import coefs_to_points
import scipy.optimize
import pandas as pd

class StereoSystem:
    '''
    A class to combine two CameraCalibrations into a stereo vision system.
    
    Parameters
    ----------
    
    calib_A,calib_B : CameraCalibration
        The two camera calibrations.
    '''
    
    def __init__(self,calib_A,calib_B):        
        self.calibs = [calib_A,calib_B]        
        
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
        
DEFAULT_MATCH_PARAMS = {'ray_sep_thresh':1e-3, # [m], separation between rays
                        'rel_size_error_thresh':0.5, # max allowable of abs(d_A-d_B)/(0.5*(d_A+d_B))
                        'min_d_for_rel_size_criteria':1e-3, # [m], value of 0.5*(d_A+d_B) below which rel_size_error_thresh is not applied
                        }
    
class Matcher:
    
    def __init__(self,df_A,df_B,stereo_system,params={}):
        
        self.df_A = df_A
        self.df_B = df_B
        self.stereo_system = stereo_system
        
        # set matcher params
        self.params = DEFAULT_MATCH_PARAMS.copy()
        for key in params:
            self.params[key] = params[key]
            
        # initialize arrays in which pairing data will be stored
        self.locs = np.zeros((len(df_A),len(df_B),3)).astype(float)
        self.errs = np.zeros((len(df_A),len(df_B))).astype(float)
        self.mask = np.ones_like(self.errs).astype(float)
        self.diameters = np.zeros_like(self.errs)
        self.diameter_diffs = np.zeros_like(self.errs)
            
    def _get_all_pairings(self):
        '''With n_A and n_B rows in df_A and df_B, compute the n_A x n_B 
        pairings (the location and error associated with each)
        '''
        
        df_A = self.df_A
        df_B = self.df_B
        
        # get the pairing positions and errors for each combination
        for aii,ai in enumerate(df_A.index):
            for bii,bi in enumerate(df_B.index):
                self.locs[aii,bii,:],self.errs[aii,bii] = self.stereo_system((df_A.loc[ai,'x'],df_A.loc[ai,'y']),(df_B.loc[bi,'x'],df_B.loc[bi,'y']))

    def _mask_on_error(self):
        '''set the mask to nan for pairings for which the error is too large
        '''
        is_bad_pos = self.errs > self.params['ray_sep_thresh']
        self.mask[is_bad_pos] = np.nan
        
    def _apply_mask(self):
        '''multiply a bunch of arrays by the mask to nan-out masked pairings
        '''
        attrs = ['locs','errs','diameter_means','diameter_diffs']
        for attr in attrs:
            setattr(self,attr,(getattr(self,attr).T*self.mask.T).T)
    
    def _get_all_diameters(self):
        '''Compute all the diameters for each pairing that is not currently
        masked'''
        df_A,df_B = self.df_A, self.df_B
        calib_A, calib_B = self.stereo_system.calibs
        # get the diameter of each object given each pairing
        d_A = np.zeros_like(self.mask) * np.nan
        d_B = np.zeros_like(self.mask) * np.nan
        for aii,ai in enumerate(df_A.index):
            d_A_px = df_A.loc[ai,'d_px']
            for bii,bi in enumerate(df_B.index):
                d_B_px = df_B.loc[bi,'d_px']
                if np.isnan(self.mask[aii,bii])==False:
                    d_A[aii,bii] = d_A_px * calc_dx((df_A.loc[ai,'x'],df_A.loc[ai,'y']),self.locs[aii,bii,:],calib_A,d_px=1,axes=None)
                    d_B[aii,bii] = d_B_px * calc_dx((df_B.loc[bi,'x'],df_B.loc[bi,'y']),self.locs[aii,bii,:],calib_B,d_px=1,axes=None)
        self.diameters_AB = np.moveaxis(np.array([d_A,d_B]),0,-1)
        self.diameter_diffs = np.diff(self.diameters_AB,axis=-1)[...,0]
        self.diameter_means = np.mean(self.diameters_AB,axis=-1)
        
    def _mask_on_diameter_difference(self):
        '''Set mask to nan for pairings for which the difference in diameter
        (as computed between the two views) is too large
        '''
        relative_size_error = np.abs(self.diameter_diffs)/self.diameter_means
        is_bad_size = (relative_size_error > self.params['rel_size_error_thresh']) & (self.diameter_means > self.params['min_d_for_rel_size_criteria'])
        self.mask[is_bad_size] = np.nan
        
    def _make_dfs(self):
        # make things DataFrames
        attrs = ['mask','errs','diameter_diffs']
        for attr in attrs:
            setattr(self,attr,pd.DataFrame(index=self.df_A.index,columns=self.df_B.index,data=getattr(self,attr)))
        
    def _find_pairs(self):
        
        mask = self.mask
        errs = self.errs
        diameter_diffs = self.diameter_diffs
        
        def _drop_nans(df):
            for i in [0,1]:
                df.dropna(how='all',axis=i,inplace=True)    
        
        # list to store the pairs, list of (ix_A,ix_B) tuples
        pairs = []
        
        # handle the easy ones first, where both row/column have just one match
        for ai in mask.index:
            if mask.loc[ai,:].sum()==1:
                bii = np.squeeze(np.argwhere(np.atleast_1d(mask.loc[ai,:])==1))
                bi = mask.columns[bii]
                if mask.loc[:,bi].sum()==1:
                    pairs.append((ai,bi))
                
        # drop the rows/column froms the dataframes
        for i in [0,1]:
            mask = mask.drop([pair[i] for pair in pairs],axis=i)
            errs = errs.drop([pair[i] for pair in pairs],axis=i)
            diameter_diffs = diameter_diffs.drop([pair[i] for pair in pairs],axis=i)
            
        # pair the ones that are each other's min for both dist and size error
        for ai in errs.index:
            
            # ai might have been removed from the index so check that it's still there
            if ai in errs.index:
                
                bi_dist = errs.loc[ai,:].idxmin()
                bi_size = diameter_diffs.loc[ai,:].idxmin()
                if bi_dist==bi_size:
                    # see if the closest match for bi is also ai
                    bi = bi_dist
                    ai_dist = errs.loc[:,bi].idxmin()
                    ai_size = diameter_diffs.loc[:,bi].idxmin()
                    if ai==ai_dist==ai_size:
                        #print(ai,bi)
                        pairs.append((ai,bi))
                        errs = errs.drop(ai,axis=0)
                        errs = errs.drop(bi,axis=1)
                        diameter_diffs = diameter_diffs.drop(ai,axis=0)
                        diameter_diffs = diameter_diffs.drop(bi,axis=1)
                        # get rid of the rows/columns with no matches
                        [_drop_nans(df) for df in [errs,diameter_diffs]]
                    
        # pair the ones that are each other's min for dist
        for ai in errs.index:
            
            # ai might have been removed from the index so check that it's still there
            if ai in errs.index:
                
                bi = errs.loc[ai,:].idxmin()
                ai_dist = errs.loc[:,bi].idxmin()
                if ai==ai_dist:
                    pairs.append((ai,bi))
                    errs = errs.drop(ai,axis=0)
                    errs = errs.drop(bi,axis=1)
                    diameter_diffs = diameter_diffs.drop(ai,axis=0)
                    diameter_diffs = diameter_diffs.drop(bi,axis=1)
                    # get rid of the rows/columns with no matches
                    [_drop_nans(df) for df in [errs,diameter_diffs]]
                    
        # Return the results
        pair_locs = []
        pair_errs = []
        for pair in pairs:
            ai,bi = pair
            aii = np.squeeze(np.argwhere(self.df_A.index==ai))
            bii = np.squeeze(np.argwhere(self.df_B.index==bi))
            pair_locs.append(self.locs[aii,bii,:])
            pair_errs.append(errs.loc[aii,bii])
            
        return pairs,pair_locs,pair_errs
        
    def match(self):
        
        # compute the pairings of all points
        self._get_all_pairings()
        
        # mask out the ones that are obviously wrong
        self._mask_on_error()
        self._get_all_diameters()
        self._mask_on_diameter_difference()
        self._apply_mask()
        
        # make the arrays dataframes
        self._make_dfs()
        
        # find the pairs
        #pairs,pair_locs,pair_errs = self._find_pairs()
        
        #return pairs,pair_locs,pair_errs
    
def find_pairs_v2(df_A,df_B,stereo_system,ray_sep_thresh=1e-3,rel_size_error_thresh=0.5,y_lims=[-np.inf,np.inf]): # y_lims=[.029,0.25]
    '''
    Given DataFrames of particles identified in two views, find which pairs of 
    identified objects in the two views are the same particle.
    
    Parameters
    ----------
    
    df_A,df_B : pandas.DataFrame
        The dataframes of the identified objects in the two views, each having
        columns x and y giving the pixel locations of the objects.
        
    stereo_system : StereoSystem
        The stereo vision system used for calculating the ray paths
        
    ray_sep_thresh : float, optional.
        The maximum separation between two rays that is acceptable for calling
        two 2D objects the same particle.
        
    rel_size_error_thresh : float, optional.
        The maximum permissible value of np.abs(d_A-d_B)/(.5*(d_A+d_B)), where
        the sizes are computed given the calculated local value of dx.
        
    y_lims : array-like of float, optional.
        The [minimum,maximum] values of the y coordinate. Pairings with y 
        values outside this range will be neglected.
        
    Returns
    ----------
    
    pairs : list of tuples
        A list of tuples, each of length 2, each giving (ix_A,ix_B), the 
        indices of the 2d particles in the pairing.
        
    pair_locs : list of np.ndarrays
        A list giving the (x,y,z) location of each pair.
        
    pair_dists : list of floats
        A list giving the error distance associated with each pair.
    '''
    
    idx_A = df_A.index.values.copy()
    idx_B = df_B.index.values.copy()
    
    calib_A,calib_B = stereo_system.calibs
    
    # get the pairing positions and errors for each combination
    pairings = np.zeros((len(df_A),len(df_B),3))
    dists = np.zeros((len(df_A),len(df_B)))
    for aii,ai in enumerate(df_A.index):
        for bii,bi in enumerate(df_B.index):
            pairings[aii,bii,:],dists[aii,bii] = stereo_system((df_A.loc[ai,'x'],df_A.loc[ai,'y']),(df_B.loc[bi,'x'],df_B.loc[bi,'y']))
    
    # mask the ones that are too far apart in location
    pair_errs = dists.copy()
    is_bad_pos = dists>ray_sep_thresh
    mask = np.ones_like(dists).astype(float)
    mask[is_bad_pos] = np.nan
    
    # # mask the ones that are outside the tank
    # is_bad_y = np.zeros_like(is_bad_pos).astype(bool)
    # is_bad_y[pairings[:,:,1]<y_lims[0]] = True
    # is_bad_y[pairings[:,:,1]>y_lims[1]] = True
    # mask[is_bad_y] = np.nan
    
    # get the diameter of each object given each pairing
    d_A = np.zeros_like(dists)
    d_B = np.zeros_like(dists)
    for aii,ai in enumerate(df_A.index):
        d_A_px = df_A.loc[ai,'d_px']
        for bii,bi in enumerate(df_B.index):
            d_B_px = df_B.loc[bi,'d_px']
            if np.isnan(mask[aii,bii])==False:
                d_A[aii,bii] = d_A_px * calc_dx((df_A.loc[ai,'x'],df_A.loc[ai,'y']),pairings[aii,bii,:],calib_A,d_px=1,axes=None)
                d_B[aii,bii] = d_B_px * calc_dx((df_B.loc[bi,'x'],df_B.loc[bi,'y']),pairings[aii,bii,:],calib_B,d_px=1,axes=None)
                
    # mask the ones which are too far apart in size
    relative_size_error = np.abs(d_A-d_B)/(.5*(d_A+d_B))
    is_bad_size = relative_size_error > rel_size_error_thresh
    mask[is_bad_size] = np.nan
    
    # apply the mask
    dists = dists*mask
    size_error = relative_size_error*mask*np.abs(d_A-d_B)
    pairings = np.moveaxis(np.moveaxis(pairings,-1,0)*mask,0,-1)
    
    # make things DataFrames
    mask = pd.DataFrame(index=idx_A,columns=idx_B,data=mask)
    dists = pd.DataFrame(index=idx_A,columns=idx_B,data=dists)
    size_error = pd.DataFrame(index=idx_A,columns=idx_B,data=size_error)

    # get rid of the rows/columns with no matches
    def drop_nans(df):
        for i in [0,1]:
            df.dropna(how='all',axis=i,inplace=True)    
    [drop_nans(df) for df in [mask,dists,size_error]]
    
    # list to store the pairs
    pairs = []
    
    # handle the easy ones first, where both row/column have just one match
    for ai in mask.index:
        if mask.loc[ai,:].sum()==1:
            bii = np.squeeze(np.argwhere(np.atleast_1d(mask.loc[ai,:])==1))
            bi = mask.columns[bii]
            if mask.loc[:,bi].sum()==1:
                #print(ai,bi)
                pairs.append((ai,bi))
            
    # drop the rows/column froms the dataframes
    for i in [0,1]:
        mask = mask.drop([pair[i] for pair in pairs],axis=i)
        dists = dists.drop([pair[i] for pair in pairs],axis=i)
        size_error = size_error.drop([pair[i] for pair in pairs],axis=i)
        
    # pair the ones that are each other's min for both dist and size error
    for ai in dists.index:
        
        # ai might have been removed from the index so check that it's still there
        if ai in dists.index:
            
            bi_dist = dists.loc[ai,:].idxmin()
            bi_size = size_error.loc[ai,:].idxmin()
            if bi_dist==bi_size:
                # see if the closest match for bi is also ai
                bi = bi_dist
                ai_dist = dists.loc[:,bi].idxmin()
                ai_size = size_error.loc[:,bi].idxmin()
                if ai==ai_dist==ai_size:
                    #print(ai,bi)
                    pairs.append((ai,bi))
                    dists = dists.drop(ai,axis=0)
                    dists = dists.drop(bi,axis=1)
                    size_error = size_error.drop(ai,axis=0)
                    size_error = size_error.drop(bi,axis=1)
                    # get rid of the rows/columns with no matches
                    [drop_nans(df) for df in [dists,size_error]]
                
    # pair the ones that are each other's min for dist
    for ai in dists.index:
        
        # ai might have been removed from the index so check that it's still there
        if ai in dists.index:
            
            bi = dists.loc[ai,:].idxmin()
            ai_dist = dists.loc[:,bi].idxmin()
            if ai==ai_dist:
                pairs.append((ai,bi))
                dists = dists.drop(ai,axis=0)
                dists = dists.drop(bi,axis=1)
                size_error = size_error.drop(ai,axis=0)
                size_error = size_error.drop(bi,axis=1)
                # get rid of the rows/columns with no matches
                [drop_nans(df) for df in [dists,size_error]]
                
    # Return the results
    pair_locs = []
    pair_dists = []
    for pair in pairs:
        ai,bi = pair
        aii = np.squeeze(np.argwhere(idx_A==ai))
        bii = np.squeeze(np.argwhere(idx_B==bi))
        pair_locs.append(pairings[aii,bii,:])
        pair_dists.append(pair_errs[aii,bii])
        
    return pairs,pair_locs,pair_dists

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

def scale_df(df,calibs,labels=['A','B']):
    '''calculate dx and bubble size for each row in a dataframe
    '''
    
    # calculate dx for each view
    for ix in df.index:
        for lab,calib in zip(labels,calibs):
            df.loc[ix,'dx_'+lab] = calc_dx((df.loc[ix,'x_'+lab],df.loc[ix,'y_'+lab]),df.loc[ix,['x','y','z']].values,calib,d_px=1,axes=None)
            
    # calculate d for each view
    for lab in labels:
        df['d_'+lab] = df['d_px_'+lab]*df['dx_'+lab]
        
    # average the d values to get d for each bubble
    d_cols = ['d_'+lab for lab in labels]
    df['d'] = np.mean(df[d_cols],axis=1)
    
    return df

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