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
from skimage.feature.texture import local_binary_pattern
import time
import sys
import random


from radiomics.featureextractor import RadiomicsFeatureExtractor



from  Util import *
import SimpleITK as sitk

from skimage import exposure

class FeatureExtraction:
    def __init__(self, arg):
        self.arg = arg
        #self.columns = []

    def process_lbp(sefl, arg):
        inputdir = arg["inputdir"]
        outputdir = arg["outputdir"]
        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]
        imageName = arg["imageName"]
        pathmasks = arg["masks"]

        base = os.path.basename(imageName)
        base = os.path.splitext(base)
        imgoname = base[0]

        imagefi = os.path.join(inputdir, imagedir, targetSet, imageName)
        maskfi = os.path.join(inputdir, pathmasks, targetSet, imgoname+'.nrrd')


        inputImage = cv2.imread(imagefi, cv2.IMREAD_GRAYSCALE)
        mask = sitk.ReadImage(maskfi)
        mask = sitk.GetArrayFromImage(mask)

        radiusl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #radiusl = [3,5,8,10]
        #radiusl = [8, 10, 12]
        #radiusl = [10]
        vect = []
        vect_names = []
        for radius in radiusl:
            nPoints = 8 * radius
            lbp = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
            xnBins = int(lbp.max() + 1)
            histogram, _ = np.histogram(lbp[np.where(mask == True) ], bins=xnBins, range=(0, xnBins))
            aux = histogram.tolist()
            vect += aux
            #print(len(aux),  radius, aux)
            vect_names += ["LBP_r"+str(radius)+"_"+str(i+1) for i in range(len(aux))]
        #print("vect", vect, vect_names, len(vect_names))
        return vect, vect_names


    def process_pyradiomics(self, arg):
        inputdir = arg["inputdir"]
        outputdir = arg["outputdir"]
        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]
        imageName = arg["imageName"]
        pathmasks = arg["masks"]

        base = os.path.basename(imageName)
        base = os.path.splitext(base)
        imgoname = base[0]

        imagefi = os.path.join(inputdir, imagedir, targetSet, imageName)
        maskfi = os.path.join(inputdir, pathmasks, targetSet, imgoname+'.nrrd')


        
        image = sitk.ReadImage(imagefi, sitk.sitkFloat32)
        mask = sitk.ReadImage(maskfi)


        """
        #begin normalization
        imagearray = sitk.GetArrayFromImage(image)
        #imagearray_eqh = exposure.equalize_adapthist(imagearray/np.max(imagearray))
        imagearray_eqh = exposure.equalize_hist(imagearray)
        #print(imagearray_eqh)
        image = sitk.GetImageFromArray(imagearray_eqh)
        #end normalization
        """


        """
        settings = {}
        settings['binWidth'] = 25
        settings['verbose'] = False
        settings['distances']  = [1, 2, 5, 10, 15]
        extractor = RadiomicsFeatureExtractor(**settings) """
        extractor = RadiomicsFeatureExtractor()
        #binWidth=20

        # Enable a filter (in addition to the 'Original' filter already enabled)
        #extractor.enableInputImageByName('LoG')
        #extractor.enableInputImages(wavelet= {'level': 2})


        #extractor.enableAllFeatures()
        #extractor.enableInputImageByName('LoG')
        
        extractor.enableImageTypes(
            Original={},
            Wavelet={},
            LoG={'sigma':[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.0]},
            LBP2D={},
            #SquareRoot={},
            #Exponential={},
            #Logarithm={},
            #Gradient={},
            )
      
        """ 
        for imageType in extractor.enabledImagetypes.keys():
            print('\t' + imageType)
        """
        
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')#19
        extractor.enableFeatureClassByName('glcm')#24
        extractor.enableFeatureClassByName('glrlm')#16   
        extractor.enableFeatureClassByName('glszm')#16
        extractor.enableFeatureClassByName('ngtdm')#5
        extractor.enableFeatureClassByName('gldm')#14
        #extractor.enableFeatureClassByName('lbp')#14

        result = extractor.execute(image, mask, label=1)
        
         
        features_names = []
        features_values = []        
        for k,v in result.items():
            #print("features_names", k)
            if k.startswith('original') or k.startswith('wavelet') or k.startswith('log') or k.startswith('lbp') or k.startswith('logarithm')  or k.startswith('exponential') or k.startswith('squareroot') or k.startswith('gradient'): 
            #if not k.startswith('general_info') or not k.startswith('diagnostics_'):
                features_names.append(k)
                v = "{:.8f}".format(v)
                #print("vec", v)
                #print("XXXXXXXXXXXXXXXXXX", v)
                features_values.append(v)
        
        return features_values, features_names

    def process(self, arg):
        inputdir = arg["inputdir"]
        outputdir = arg["outputdir"]
        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]
        imageName = arg["imageName"]
        pathmasks = arg["masks"]

        base = os.path.basename(imageName)
        base = os.path.splitext(base)
        imgoname = base[0]



        lbpfeat, lpbnames = self.process_lbp(arg)
        radfeat, radbnames = self.process_pyradiomics(arg)

        features_names  =  ["image"]+lpbnames+radbnames+["target"]
        features_values  =  [imageName]+lbpfeat+radfeat+[targetSet]
        #print(features_names, features_values)
        return features_names, features_values

    def execute(self):
        inputdir = self.arg["inputdir"]
        outputdir = self.arg["outputdir"]
        imagedir = self.arg["imagedir"]
        masks = self.arg["masks"]

        
        Util.makedir(outputdir)

        arg = []
        for targ in os.listdir(inputdir + '/'+imagedir):
            Util.makedir(os.path.join(inputdir, masks, targ))
            for imageName in os.listdir(inputdir + '/'+imagedir+'/' + targ):
                #print("imageName", imageName)
                dat = self.arg.copy()
                dat["targetSet"] = targ
                dat["imageName"] = imageName
                arg.append(dat)

        ncpus = multiprocessing.cpu_count()-1
        pool = Pool(processes=ncpus)
        rr = pool.map(self.process, arg)
        pool.close()
        
        #print(rr)
        
        columns = rr[0][0]
        dataset = []
        for rs in rr:
            #print(rs[0])
            #print(rs[1])
            dataset.append(rs[1])

        df = pd.DataFrame(data=dataset)
        df.columns = columns

        df.to_csv(os.path.join(outputdir, self.arg["file"]), index=False)




if __name__ == "__main__":
    """ 
    arg =   {
                "inputdir":"../../../../data/bone",
                "outputdir":"../../../../data/bone/build/csv",
                "imagedir":"images",
                "masks":"masks",
                "file":"data_v3_r3.csv"
            }
    model = FeatureExtraction(arg)
    model.execute()
    """

    
    arg =   {
                "inputdir":"../../../../data/bone/ds2",
                "outputdir":"../../../../data/bone/ds2/build/csv",
                "imagedir":"images",
                "masks":"masks",
                "file":"data_v4.csv"
            }
    model = FeatureExtraction(arg)
    model.execute()
