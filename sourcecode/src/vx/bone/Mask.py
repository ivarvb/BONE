#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""
import multiprocessing

from multiprocessing import Pool, Manager, Process, Lock
#from sourcecode.src.vx.pclas.Description import Description

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.twodim_base import mask_indices
import pandas as pd
import os
import time
import sys
import random


from  Util import *
import SimpleITK as sitk

class Mask:
    def __init__(self, arg):
        self.arg = arg
        #self.columns = []

    def process_make_masks(self, arg):
        inputdir = arg["inputdir"]
        outputdir = arg["outputdir"]
        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]
        imageName = arg["imageName"]
        pathmasks = arg["masks"]

        filein = os.path.join(inputdir, imagedir, targetSet, imageName)
        pathou = os.path.join(inputdir, pathmasks, targetSet, )
        
        base = os.path.basename(imageName)
        base = os.path.splitext(base)
        imgoname = base[0]

        print("filein", filein)
        #image = sitk.ReadImage(filein)
        image = cv2.imread(filein, cv2.IMREAD_GRAYSCALE)
        print(image)
        #image_array = sitk.GetArrayViewFromImage(image)
        #print(image_array)
        
        im_size = image.shape

        mask = np.zeros(im_size, dtype=int)
        print("im_size", im_size, image.shape)
        mask[np.where(image != 0)] = 1
        print(mask)

        sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(pathou, imgoname+'.nrrd'), True)
        cv2.imwrite(os.path.join(pathou, imgoname+'.jpg'), mask*255)

    def execute(self):
        inputdir = self.arg["inputdir"]
        outputdir = self.arg["outputdir"]
        imagedir = self.arg["imagedir"]
        masks = self.arg["masks"]

        

        print("BEGIN: ")
        arg = []
        for targ in os.listdir(inputdir + '/'+imagedir):
            Util.makedir(os.path.join(inputdir, masks, targ))
            for imageName in os.listdir(inputdir + '/'+imagedir+'/' + targ):
                print("imageName", imageName)
                #Util.makedir(outputdir + '/' + targ)
                #print(self.arg)
                dat = self.arg.copy()
                dat["targetSet"] = targ
                dat["imageName"] = imageName
                arg.append(dat)

        ncpus = multiprocessing.cpu_count()-1
        dataset = pd.DataFrame()
        pool = Pool(processes=ncpus)
        rr = pool.map(self.process_make_masks, arg)
        pool.close()
        
         
        for rs in rr:
            if len(dataset)==0:
                dataset = rs[0]+rs[1]
            else:
                dataset = dataset + rs[0] + rs[1]
            #print(dataset)

        columns = ["image","tilepercentage","loc1","loc2","loc3","loc4","idseg","target"]

        df = pd.DataFrame(data=dataset)
        df.columns = columns

        df.to_csv(self.arg["outputdir"]+"/"+self.arg["file"], index=False)
        print("END: ")
        





if __name__ == "__main__":
    
    arg =   {
                "inputdir":"/mnt/sda6/software/projects/data/bone",
                "imagedir":"images",
                "outputdir":"/mnt/sda6/software/projects/data/bone/masks",
                "masks":"masks"
            }
    model = Mask(arg)
    model.execute()
    
    