# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:47:46 2021

@author: druth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import joblib
import time

from .stereo_system import find_epipolar_line_given_otherpx
from ..camera.camera import calc_dx

DEFAULT_MATCH_PARAMS = {'ray_sep_thresh':1e-3, # [m], separation between rays
                        'rel_size_error_thresh':0.5, # max allowable of abs(d_A-d_B)/(0.5*(d_A+d_B))
                        'min_d_for_rel_size_criteria':2e-3, # [m], value of 0.5*(d_A+d_B) below which rel_size_error_thresh is not applied
                        'epipolar_dist_thresh_px':None}
class Matcher:
    '''
    Match the objects that are detected in two images with an instance of a 
    StereoSystem.
    
    Parameters
    ----------
    df_A, df_B : pd.DataFrame
        Dataframes containing information about the detected 2-d objects in the 
        two views. Must contain the columns x and y, which give the pixel 
        location of the object, and d_px, which is the diameter of the object
        in pixels.
        
    stereo_system : stereo.stereo_system.StereoSystem
        The stereo system for the two views.
        
    params : dict, optional
        The parameters for the matching. DEFAULT_MATCH_PARAMS is used by 
        default, and key-value pairs from params overwrite the default values.
        
        Key-value pairs are:
            ray_sep_thresh : float
                The maximum allowable triangulation error (the shortest
                distance between two light rays paired as the same object) in
                meters.
            rel_size_error_thresh : float
                The max allowable value of abs(d_A-d_B)/(0.5*(d_A+d_B)), where
                d_A and d_B are the object sizes as determined between the two
                views. This filteirng is only applied to pairings for which the
                average of the two sizes is greater than 
                min_d_for_rel_size_criteria.
            min_d_for_rel_size_criteria : float
                The minimum object size of a pairing for which the relative 
                size filtering is applied, in meters.
            epipolar_dist_thresh_px : float or None
                The maximum allowable distance an object in image B can be from
                the epipolar line of its paired image from image A. This can be
                used to speed up the pairing so long as view B has a working
                inverse interpolator. If None, this filtering is not applied 
                during the matching.
        
    frame : None or int, optional
        If not None, df_A and df_B are filtered to only include rows for which
        the "frame" column is equal to frame (to avoid filtering the dataframe
        manually as the class is initialized.)
    '''
    
    def __init__(self,df_A,df_B,stereo_system,params=None,frame=None):
        
        if frame is not None:
            df_A = df_A[df_A['frame']==frame]
            df_B = df_B[df_B['frame']==frame]
        
        self.df_A = df_A.copy()
        self.df_B = df_B.copy() 
        self.stereo_system = stereo_system

        # set matcher params
        self.params = DEFAULT_MATCH_PARAMS.copy()
        if params is not None:
            for key in params:
                self.params[key] = params[key]
            
        # initialize arrays in which pairing data will be stored
        self.locs = np.zeros((len(df_A),len(df_B),3)).astype(float) * np.nan
        self.errs = np.zeros((len(df_A),len(df_B))).astype(float) * np.nan
        self.mask = np.ones_like(self.errs).astype(float)
        self.diameters = np.zeros_like(self.errs) * np.nan
        self.diameter_diffs = np.zeros_like(self.errs) * np.nan
        
        # where to store the (ix_A,ix_B) pairs and resulting paired DataFrame
        self.pairs = []
        self.pairs_i = []
        self.df = pd.DataFrame(columns=['x','y','z','frame','err'])
        
    def show_state(self,):
        '''Make a figure with plots showing detected objects in each view, and 
        lines connecting paired objects.        
        '''
        
        fig,axs = plt.subplots(1,2,figsize=(9,4))
        
        for df,ax,calib in zip([self.df_A,self.df_B],axs,self.stereo_system.calibs):
            ax.scatter(df['x'],df['y'],color='gray',)
            calib.set_axes_lims(ax)
        
        for ai,a in enumerate(self.df_A.index):
            for bi,b in enumerate(self.df_B.index):
                if ~np.isnan(self.mask[ai,bi]):
                    cp = ConnectionPatch(xyA=[self.df_A.loc[a,'x'],self.df_A.loc[a,'y']],
                                         xyB=[self.df_B.loc[b,'x'],self.df_B.loc[b,'y']],
                                         coordsA='data',
                                         coordsB='data',
                                         axesA=axs[0],
                                         axesB=axs[1],
                                         alpha=0.5)
                    axs[1].add_patch(cp)
                    
        fig.tight_layout()
        
    def _mask_on_dist_to_epipolar(self,n_points_epipolar=51):
        '''see which bubbles in image B are close to the epipolar line for each 
        bubble in image A
        '''
        
        df_A = self.df_A
        df_B = self.df_B
        
        calib_A = self.stereo_system.calibs[0]
        calib_B = self.stereo_system.calibs[1]
        
        y_lims = self.stereo_system.lims['y']
        
        not_to_mask = np.zeros_like(self.mask)
        dists_to_epipolar = np.zeros_like(self.mask)*np.nan
        
        for aii,ai in enumerate(df_A.index):
            
            xy_A = np.array([df_A.loc[ai,'x'],df_A.loc[ai,'y']])
            
            # possible locations in image B
            xy_B_curve = find_epipolar_line_given_otherpx(calib_A,calib_B,xy_A,np.linspace(y_lims[0],y_lims[1],n_points_epipolar))
            
            # distance from each df_B bubble to each point on epipolar line
            dist_points = np.sqrt(np.subtract.outer(df_B['x'].values,xy_B_curve[:,0])**2 + np.subtract.outer(df_B['y'].values,xy_B_curve[:,1])**2)
            
            # closest distance to epipolar line
            min_dist = np.min(dist_points,axis=1)
            dists_to_epipolar[aii,:] = min_dist
            close_enough = min_dist < self.params['epipolar_dist_thresh_px']
            ai_close_enough = np.squeeze(np.argwhere(close_enough))
                
            not_to_mask[aii,ai_close_enough] = 1
        
        self.mask[not_to_mask==0] = np.nan
        self.dists_to_epipolar = dists_to_epipolar
                            
    def _get_all_pairings(self,use_epipolar_distance=False):
        '''With n_A and n_B rows in df_A and df_B, compute the n_A x n_B 
        pairings (the location and error associated with each)
        '''
        
        df_A = self.df_A
        df_B = self.df_B
        
        # get the pairing positions and errors for each combination
        for aii,ai in enumerate(df_A.index):
            for bii,bi in enumerate(df_B.index):
                if np.isnan(self.mask[aii,bii])==False:
                    self.locs[aii,bii,:],self.errs[aii,bii] = self.stereo_system((df_A.loc[ai,'x'],df_A.loc[ai,'y']),(df_B.loc[bi,'x'],df_B.loc[bi,'y']))

    def _mask_on_error(self):
        '''set the mask to nan for pairings for which the error is too large
        '''
        is_bad_err = self.errs > self.params['ray_sep_thresh']
        self.mask[is_bad_err] = np.nan
        
    def _mask_on_pos(self):
        '''set the mask to nan for pairings outside the StereoSystem's limits
        '''
        for ai,axis in enumerate(['x','y','z']):
            is_bad_axis = np.zeros_like(self.errs).astype(bool)
            is_bad_axis[self.locs[:,:,ai]<self.stereo_system.lims[axis][0]] = True
            is_bad_axis[self.locs[:,:,ai]>self.stereo_system.lims[axis][1]] = True
            self.mask[is_bad_axis] = np.nan
        
    def _apply_mask(self):
        '''multiply a bunch of arrays by the mask to nan-out masked pairings
        '''
        attrs = ['locs','errs','diameters','diameter_diffs']
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
        dx_A = np.zeros_like(self.mask) * np.nan
        dx_B = np.zeros_like(self.mask) * np.nan
        for aii,ai in enumerate(df_A.index):
            d_A_px = df_A.loc[ai,'d_px']
            for bii,bi in enumerate(df_B.index):
                d_B_px = df_B.loc[bi,'d_px']
                if np.isnan(self.mask[aii,bii])==False:
                    dx_A[aii,bii] = calc_dx((df_A.loc[ai,'x'],df_A.loc[ai,'y']),self.locs[aii,bii,:],calib_A,d_px=1,axes=None)
                    dx_B[aii,bii] = calc_dx((df_B.loc[bi,'x'],df_B.loc[bi,'y']),self.locs[aii,bii,:],calib_B,d_px=1,axes=None)
                    d_A[aii,bii] = d_A_px * dx_A[aii,bii]
                    d_B[aii,bii] = d_B_px * dx_B[aii,bii]
        self.dx_AB = np.moveaxis(np.array([dx_A,dx_B]),0,-1) # [ai,bi,which_view]
        self.diameters_AB = np.moveaxis(np.array([d_A,d_B]),0,-1) # [ai,bi,which_view]
        self.diameter_diffs = np.abs(np.diff(self.diameters_AB,axis=-1))[...,0]
        self.diameters = np.mean(self.diameters_AB,axis=-1)
        
    def _mask_on_diameter_difference(self):
        '''Set mask to nan for pairings for which the difference in diameter
        (as computed between the two views) is too large
        '''
        relative_size_error = self.diameter_diffs/self.diameters
        is_bad_size = (relative_size_error > self.params['rel_size_error_thresh']) & (self.diameters > self.params['min_d_for_rel_size_criteria'])
        self.mask[is_bad_size] = np.nan
        
    def _find_pairs(self):
        
        mask = self.mask
        errs = self.errs
        diameter_diffs = self.diameter_diffs
        df_A = self.df_A
        df_B = self.df_B
                
        def _nan_pairs(_1,_2):
            '''
            Mask a pairing once it's established, so it's not again added as a 
            pair in a later round of pairing.
            '''
            #mask[aii,bii] = np.nan
            pass
        
        # list to store the pairs, list of (ix_A,ix_B) tuples
        pairs_i = [[np.nan,np.nan]] # [aii,bii]
        
        # handle the easy ones first, where both row/column have just one match
        for aii in range(len(df_A)):
            if np.nansum(mask[aii,:])==1:
                bii = np.squeeze(np.argwhere(np.atleast_1d(mask[aii,:])==1))
                if np.nansum(mask[:,bii])==1:
                    pairs_i.append((aii,bii))
                    _nan_pairs(aii,bii)
        
            
        # pair the ones that are each other's min for both dist and size error
        for aii in [aii for aii in range(len(df_A)) if not (aii in np.atleast_2d(pairs_i)[:,0]) and ~np.all(np.isnan(errs[aii,:]))]:
            
            bii_dist = np.squeeze(np.nanargmin(errs[aii,:])) # bii index of smallest distance error
            bii_size = np.squeeze(np.nanargmin(diameter_diffs[aii,:])) # bii index of smallest diameter error
            if bii_dist==bii_size and not (bii_dist in np.array(pairs_i)[:,1]):
                bii = bii_dist
                aii_dist = np.squeeze(np.nanargmin(errs[:,bii]))
                aii_size = np.squeeze(np.nanargmin(diameter_diffs[:,bii]))
                if aii==aii_dist==aii_size and ~np.isnan(mask[aii,bii]):
                    pairs_i.append((aii,bii))
                    _nan_pairs(aii,bii)
                                        
        # pair the ones that are each other's min for both dist and size error
        for aii in [aii for aii in range(len(df_A)) if not (aii in np.atleast_2d(pairs_i)[:,0]) and ~np.all(np.isnan(errs[aii,:]))]:
            
            # bii of smallest distance error of aii
            bii = np.squeeze(np.nanargmin(errs[aii,:]))
            
            # aii of smallest distance error of bii
            aii_of_bii = np.squeeze(np.nanargmin(errs[:,bii]))
            
            # if these two are the same, make a pair
            if aii==aii_of_bii and ~np.isnan(mask[aii,bii]) and not (bii in np.atleast_2d(pairs_i)[:,1]):
                pairs_i.append((aii,bii))
                _nan_pairs(aii,bii)
                                    
        # make a list of the pairs by the dataframe index
        pairs = []
        pairs_i = [p for p in pairs_i if ~np.isnan(p[0])]
        for pair_i in pairs_i:
            aii,bii = pair_i
            pairs.append((df_A.index[aii],df_B.index[bii]))
            
        self.pairs = pairs # indicies of DataFrames
        self.pairs_i = pairs_i # indicies of arrays
            
        return pairs, pairs_i
    
    def _pairs_to_df(self,pairs=None):
        
        if pairs is None:
            pairs = self.pairs
            
        # put the results (location, error, 2d properties) in a dataframe
        props_2d = ['x','y','d_px','orientation','eccentricity','perimeter_px','min_axis_px','maj_axis_px']
        j = 0
        for pair in pairs:
            aii = np.squeeze(np.argwhere(self.df_A.index==pair[0]))
            bii = np.squeeze(np.argwhere(self.df_B.index==pair[1]))
            self.df.loc[j,['x','y','z']] = self.locs[aii,bii,:]
            self.df.loc[j,'frame'] = self.df_A.loc[pair[0],'frame']
            self.df.loc[j,'err'] = self.errs[aii,bii]
            if self.params['epipolar_dist_thresh_px'] is not None:
                self.df.loc[j,'dist_to_epipolar'] = self.dists_to_epipolar[aii,bii]
            for suffix,df_use,i in zip(['A','B'],[self.df_A,self.df_B],[0,1]):
                for prop in props_2d:
                    self.df.loc[j,prop+'_'+suffix] = df_use.loc[pair[i],prop]
                self.df.loc[j,'ix'+suffix] = pair[i]
                self.df.loc[j,'dx_'+suffix] = self.dx_AB[aii,bii,i]
                self.df.loc[j,'d_'+suffix] = self.df.loc[j,'d_px_'+suffix] * self.df.loc[j,'dx_'+suffix]
            j = j+1
        if len(self.df)>0:
            self.df['d'] = self.df[['d_A','d_B']].mean(axis=1)
        
        return self.df
        
    def match(self):
        '''Run the matching procedure. First, filters out pairings based on the
        distance to the epipolar lines, then masks on other quantities, then
        finds the pairings, then puts results into a DataFrame, which is 
        returned.
        '''
        
        # mask based on the epipolar distances
        if self.params['epipolar_dist_thresh_px'] is not None:
            self._mask_on_dist_to_epipolar()
        
        # compute the pairings of all points not masked
        self._get_all_pairings()
        
        # mask based on the error, diameter difference, and position
        self._mask_on_error()
        self._mask_on_pos()
        self._get_all_diameters()
        self._mask_on_diameter_difference()
        
        # apply the mask
        self._apply_mask()
        
        # find the pairs
        _ = self._find_pairs()
        _ = self._pairs_to_df()
        
        return self.df
    
def match_multiple_frames(df_A,df_B,ss,frames=None,params={},n_threads=1):
    '''
    Match multiple frames; same syntax as Matcher, but with frames as a
    list-like.
    
    Parameters
    ----------

    df_A, df_B : pd.DataFrame
        Dataframes containing information about the detected 2-d objects in the 
        two views. Must contain the columns x and y, which give the pixel 
        location of the object, and d_px, which is the diameter of the object
        in pixels. Must contain the column 'frame', giving the frame in which
        each object is detected. Frame x in df_A must correspond to frame x in
        df_B.
        
    stereo_system : stereo.stereo_system.StereoSystem
        The stereo system for the two views.
        
    params : dict, optional
        The parameters for the matching. DEFAULT_MATCH_PARAMS is used by 
        default, and key-value pairs from params overwrite the default values.
        
        Key-value pairs are:
            ray_sep_thresh : float
                The maximum allowable triangulation error (the shortest
                distance between two light rays paired as the same object) in
                meters.
            rel_size_error_thresh : float
                The max allowable value of abs(d_A-d_B)/(0.5*(d_A+d_B)), where
                d_A and d_B are the object sizes as determined between the two
                views. This filteirng is only applied to pairings for which the
                average of the two sizes is greater than 
                min_d_for_rel_size_criteria.
            min_d_for_rel_size_criteria : float
                The minimum object size of a pairing for which the relative 
                size filtering is applied, in meters.
            epipolar_dist_thresh_px : float or None
                The maximum allowable distance an object in image B can be from
                the epipolar line of its paired image from image A. This can be
                used to speed up the pairing so long as view B has a working
                inverse interpolator. If None, this filtering is not applied 
                during the matching.
        
    frames : list-like or None
        The frames for which to perform the matching. If None, it is taken as
        all the unique frames in df_A.
    
    '''
    if frames is None:
        frames = df_A['frames'].unique().values
    
    def match_frame(f):
        print('...matching frame '+str(f))
        m = Matcher(df_A.copy(),df_B.copy(),ss,frame=f,params=params)
        return m.match()
    
    if n_threads>1:
        print('Attempting to match '+str(len(frames))+' frames in parallel with '+str(n_threads)+' jobs...')
        t1 = time.time()
        dfs_3d = joblib.Parallel(n_threads)(joblib.delayed(match_frame)(f) for f in frames)
        print('...completed in '+'{:0.1f}'.format(time.time()-t1)+' s!')
    else:
        dfs_3d = []
        for fi,f in enumerate(frames):
            dfs_3d.append(match_frame(f))
    return pd.concat(dfs_3d,ignore_index=True)