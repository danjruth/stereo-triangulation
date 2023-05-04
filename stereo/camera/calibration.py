# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:54:11 2021

@author: druth
"""

import matplotlib.pyplot as plt
import skimage.transform
import scipy.ndimage
import numpy as np
import pandas as pd

def _refine_click_point_auto(im,click,box_size=10,rescale_factor=5):
    '''
    Automatically refine the dot location by selecting the darkest point in the
    filtered original image that is within box_size of the manually-clicked
    point.
    '''
    
    # get, rescale, and filter the image region around the clicked point
    point_int = np.round(click).astype(int)    
    im_region = im[point_int[1]-box_size:point_int[1]+box_size,point_int[0]-box_size:point_int[0]+box_size]
    im_region_rescale = skimage.transform.rescale(im_region,rescale_factor,order=0)
    im_region_rescale_filt = scipy.ndimage.gaussian_filter(im_region_rescale,rescale_factor)
    #im_region_rescale_filt = im_region_rescale
    
    # refine the point and convert back to original scaling
    pt_refined_region_rescale = np.flip(np.unravel_index(im_region_rescale_filt.argmin(), im_region_rescale_filt.shape)) # x,y
    
    pt_refined_region = pt_refined_region_rescale/rescale_factor # x,y    
    pt_refined = pt_refined_region + point_int - box_size
    
    return pt_refined

def _refine_click_point_manual(im,click,box_size=10,):
    '''
    Let the user refine a clicked point manually by displaying a blown-up view
    of the region of the image near the clicked point
    '''
    
    fig,ax = plt.subplots()
    ax.imshow(im,cmap='gray') # ,vmin=0,vmax=255    
    ax.set_axis_off()
    ax.set_xlim(click[0]-box_size,click[0]+box_size)
    ax.set_ylim(click[1]+box_size,click[1]-box_size)
    point = np.array(plt.ginput(timeout=-1,n=1))[0]
    plt.close(fig)
    return point

def input_calib_points(folder,fname,yval_mm,im_extension='.tif',rot=False,invert=False,refine='auto',box_size=4,vmin=-30,vmax=5,highpass_size=9,lowpass_size=1):
    '''
    Manually click on points in a calibration image and enter their
    coordinates.
    '''
    
    # read in the image
    im = plt.imread(folder+fname+im_extension).astype(float)
    if rot:
        im = np.rot90(im)
    if invert:
        im = np.max(im)-im
        
    im = im - scipy.ndimage.gaussian_filter(im,highpass_size)
    im = scipy.ndimage.gaussian_filter(im,lowpass_size)
    
    # calculate the value of y
    y = float(yval_mm)/1000
    
    print('Click on the points in a grid.')
    print('Start at the bottom left, then work right then up.')
    
    # have the user click on the points
    fig,ax=plt.subplots(figsize=(13,11))
    ax.imshow(im,cmap='gray',vmin=vmin,vmax=vmax) # ,vmin=0,vmax=255    
    ax.set_axis_off()
    fig.tight_layout()
    points = np.array(plt.ginput(timeout=-1,n=-1))
    ax.plot(points[:,0],points[:,1],'x',color='r')
    
    # refine the points
    if refine=='automatic' or refine=='auto':
        points = np.array([_refine_click_point_auto(im,pt,box_size=box_size) for pt in points])    
    elif refine=='manual':
        points = np.array([_refine_click_point_manual(im,pt,box_size=box_size) for pt in points])
                
    ax.plot(points[:,0],points[:,1],'x',color='b')
    plt.show()
    fig.canvas.draw()
    plt.show()
    
    # store the x and y locs in a dataframe
    df = pd.DataFrame()
    df['x'] = points[:,0]
    df['y'] = points[:,1]
        
    # get the corresponding X and Z locs then store the object locations
    x = np.array(input('Enter x values in cm, separated by spaces: ').split()).astype(float)
    z = np.array(input('Enter z values in cm, separated by spaces: ').split()).astype(float)    
    df['X'] = np.array(list(x)*len(z))*0.01
    df['Z'] = np.repeat(z,len(x))*0.01
    df['Y'] = y
    
    # save the dataframe
    fpath_save = folder+'y_'+str(yval_mm)+'mm.pkl'
    df.to_pickle(fpath_save)
    
    return df