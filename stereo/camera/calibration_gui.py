# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:43:43 2021

@author: druth
"""

import numpy as np
import matplotlib.pyplot as plt
from stereo.camera import calibration, CameraCalibration
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import Rectangle, Ellipse
import os
import pandas as pd
import scipy.ndimage

class CameraCalibrationGUI:
    '''
    A graphical user interface to carry out the camera calibration for one
    camera. The calibration images should be stored as followed:
        
        BASE_FOLDER\
            CAMERA_A\
                (images of the calibration target at various y positions as
                 viewed by the camera named CAMERA_A, each named like
                 y_xmm.tif, where x is the y position in mm)
            CAMERA_B\
                (images of the calibration target at various y positions as
                 viewed by the camera named CAMERA_B, each named like
                 y_xmm.tif, where x is the y position in mm)
                
    Call this GUI once for each camera, specifying the path to BASE_FOLDER as
    the first argument and the name of the camera as the second.
    
    The code will find all the calibration images listed in
    BASE_FOLDER\camera_name\ and initially list them under "INCOMPLETE".
    
    For each image to calibrate with, enter the filename (without the
    extension) and the corresponding y position (in mm), and hit "Calibrate".
    
    Starting with the bottom left of the image (should be most negative x and 
    z calibration points), click on a grid of the calibration points going from
    left to right and then bottom to top, typically ~25 points per image. The 
    points will be refined to the darkest part of the image within 10 pixels of
    the clicked point.
    
    Then enter the x values and z values, in cm, in order, and hit "Done". This
    saves the clicked image points and corresponding physical points in a
    DataFrame saved at BASE_FOLDER\camera_name\image_name.pkl and moves the
    camera name to the "COMPLETE" list.
    
    To aggregate the data from the images in the "COMPLETE" list, hit "Create
    calibration object", which aggreates them into a DataFrame saved at
    BASE_FOLDER\camera_name.pkl
    '''
    
    def __init__(self,base_folder,camera_name,extension='.tiff'):
        
        self.extension = extension
        
        self.base_folder = base_folder
        self.camera_name = camera_name
        self.folder = base_folder + r'\\' + camera_name + r'\\'
        self.fig = plt.figure(figsize=(13,8))     
        fig = self.fig
        self.ax_im = fig.add_axes([0.01,0.01,0.6,0.98])
        self.ax_folder = fig.add_axes([0.62,0.8,0.37,0.19])
        self.ax_fnamebox = fig.add_axes([0.62, 0.74, 0.2, 0.04])
        self.ax_ystrbox = fig.add_axes([0.83, 0.74, 0.09, 0.04])
        self.ax_calibratebox = fig.add_axes([0.93, 0.74, 0.06, 0.04])
        self.ax_xbox = fig.add_axes([0.7, 0.68, 0.22, 0.04])
        self.ax_zbox = fig.add_axes([0.7, 0.63, 0.22, 0.04])
        self.ax_donebox = fig.add_axes([0.93, 0.63, 0.06, 0.09])
        self.ax_createcalibbox = fig.add_axes([0.62, 0.01, 0.37, 0.15])
        
        self.ft = self.fig.text(0.02,0.98,'Enter filename and y (mm)',horizontalalignment='left',verticalalignment='top',fontsize=14,color='k')
        
        # validation axes
        self.axs_val = [[None,None],[None,None],]
        for i in range(2):
            for j in range(2):
                self.axs_val[i][j] = fig.add_axes([0.66+0.2*i,0.22+0.23*j,0.13,0.15])
        self.axs_val = np.array(self.axs_val)

        self.current_fname = ''
        
        # file name
        self.box_fname = TextBox(self.ax_fnamebox, '')
        self.box_fname.on_submit(self.update_fname)
        self.box_fname.set_val('filename')  # Trigger `submit` with the initial string.        
        self.submitfname = Button(self.ax_calibratebox, 'Calibrate', hovercolor='0.975')
        self.submitfname.on_clicked(self.submit_on_clicked_fcn)
        self.box_ystr = TextBox(self.ax_ystrbox, '')
        self.box_ystr.on_submit(self.update_ystr)
        self.box_ystr.set_val('y [mm]')  # Trigger `submit` with the initial string.        
        
        # x values, y values
        self.box_xvals = TextBox(self.ax_xbox, '$x$ values [cm]')
        self.box_xvals.on_submit(self.update_xvals)
        self.box_xvals.set_val('')  # Trigger `submit` with the initial string.        
        self.box_zvals = TextBox(self.ax_zbox, '$z$ values [cm]')
        self.box_zvals.on_submit(self.update_zvals)
        self.box_zvals.set_val('')
        self.box_done = Button(self.ax_donebox, 'Done', hovercolor='0.975')
        self.box_done.on_clicked(self.create_df)
        
        # create calibration
        self.box_createcalib = Button(self.ax_createcalibbox, 'Create calibration object', hovercolor='0.975')
        self.box_createcalib.on_clicked(self.create_calibration)
        
        self.update_folder_view(None)
        
    def message(self,text):
        self.ft.set_text(text)
        self.fig.canvas.draw_idle()

    def update_fname(self,fname):
        self.current_fname = fname
        
    def update_ystr(self,ystr):
        self.ystr = ystr
        
    def update_xvals(self,xvals):
        self.xvals = xvals
        
    def update_zvals(self,zvals):
        self.zvals = zvals
        
    def submit_on_clicked_fcn(self,fname):
        self.input_calib_points()
    
    def update_folder_view(self,_):
        
        im_files = [f[:-len(self.extension)] for f in os.listdir(self.folder) if f[-len(self.extension):]==self.extension]
        self.complete_files = [f[:-4] for f in os.listdir(self.folder) if f[-4:]=='.pkl']
        self.incomplete_files = [f for f in im_files if f not in self.complete_files]
        
        self.ax_folder.clear()
        self.ax_folder.set_xticks([])
        self.ax_folder.set_yticks([])
        
        s_complete = 'COMPLETE'
        for c in self.complete_files:
            s_complete = s_complete+'\n'+c
            
        s_incomplete = 'INCOMPLETE'
        for c in self.incomplete_files:
            s_incomplete = s_incomplete+'\n'+c
            
        self.ax_folder.text(0.05,0.95,s_incomplete,color='r',transform=self.ax_folder.transAxes,horizontalalignment='left',verticalalignment='top')
        self.ax_folder.text(0.55,0.95,s_complete,color='g',transform=self.ax_folder.transAxes,horizontalalignment='left',verticalalignment='top')

    def input_calib_points(self,rot=False,invert=False,refine='automatic',box_size=10,):
        '''
        Manually click on points in a calibration image and enter their
        coordinates.
        '''
        
        # read in the image
        im = plt.imread(self.folder+self.current_fname+self.extension).astype(float)
        self.im_shape = np.shape(im)
        
        # highpass filter
        im = im - scipy.ndimage.gaussian_filter(im,(5,5))
        
        # calculate the value of y
        
        self.message('Click on the points in a grid (bottom left, work right then up)')
        #print('Start at the bottom left, then work right then up.')
        
        # have the user click on the points
        ax = self.ax_im
        ax.clear()
        ax.imshow(im,cmap='gray',vmin=-30,vmax=5) # ,vmin=0,vmax=255    
        ax.set_axis_off()
        ax.set_title(self.current_fname)
        #plt.sca(ax)
        #self.fig.canvas.draw()
        fig_new,ax_new = plt.subplots()
        ax_new.imshow(im,cmap='gray',vmin=-30,vmax=5)
        #plt.sca(ax_new)
        plt.pause(1)
        print('start clicking on points...')
        pts_click = plt.ginput(n=-1,timeout=-1)
        points = np.array(pts_click)
        print('...done!')
        plt.close(fig_new)
        
        self.points = np.array([calibration._refine_click_point_auto(im,pt,box_size=box_size) for pt in points])          
        ax.plot(self.points[:,0],self.points[:,1],'x',color='b')
        self.fig.canvas.draw_idle()
        
        self.message('Enter x and z values')
        
    def create_df(self,_):
        
        # store the x and y locs in a dataframe
        df = pd.DataFrame()
        df['x'] = self.points[:,0]
        df['y'] = self.points[:,1]
            
        # get the corresponding X and Z locs then store the object locations
        x = np.array(self.xvals.split()).astype(float)
        z = np.array(self.zvals.split()).astype(float)    
        df['X'] = np.array(list(x)*len(z))*0.01
        df['Z'] = np.repeat(z,len(x))*0.01
        df['Y'] = float(self.ystr)/1000
        
        self.df = df
    
        fpath_save = self.folder+self.current_fname+'.pkl'
        df.to_pickle(fpath_save)
        self.update_folder_view(None)
        self.box_xvals.set_val('') 
        self.box_zvals.set_val('')       
        
    def create_calibration(self,_):
        
        # get the dataframes and save the resulting CameraCalibration
        dfs = pd.concat([pd.read_pickle(self.folder+fname+'.pkl') for fname in self.complete_files])
        self.calib = CameraCalibration(dfs[['X','Y','Z']],dfs[['x','y']],self.im_shape,)        
        
        # plot the validation
        [ax.clear() for ax in self.axs_val.flatten()]
        self.calib.plot_known_vs_predicted(self.axs_val)
        for ax in self.axs_val.flatten():
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)
            ax.get_legend().remove()
        
        # save it
        fpath_save = self.base_folder+self.camera_name + '.pkl'
        self.calib.to_dict(fpath_save=fpath_save)
        self.message('Saved CameraCalibration to '+fpath_save)