#!/usr/bin/env python

from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import warnings

def run_demo(dataDir, dataType, annDir, annZipFile, annFile, annURL, capAnnFile):
    '''
    Runs our custom cocoapi demo
    takes in all fields from data-params.json found in config folder
    
    returns nothing
    saves demo picture to src/data
    '''
    warnings.filterwarnings("ignore")

    # now check if the annotations have been properly downloaded
    # NOTE: this isn't needed anymore due to the nature of the /teams directory
    # TODO: re-add imports if this func is required again
    #download_annotations(dataDir, dataType, annDir, annZipFile, annFile, annURL)

    print('Initializing COCO api annotations...')
    # initialize COCO api
    coco = COCO(annFile)

    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps = COCO(annFile)

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['cat']);
    imgIds = coco.getImgIds(catIds=catIds);
    print('Finished loading COCO api annotations...')

    # now plot the imagery
    plt.figure(figsize=(20,10))
    columns = 4

    for i in range(len(imgIds[:12])):
        j = coco.loadImgs(imgIds[i])[0]
        I = io.imread(j['coco_url'])
        plt.subplot(3, columns, i + 1)
        plt.imshow(I)

        annIds = coco.getAnnIds(imgIds=j['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        plt.axis('off')

        annIds = coco_caps.getAnnIds(imgIds=j['id']);
        anns = coco_caps.loadAnns(annIds)
        plt.subplot(3,columns,i+1).set_title(anns[0]['caption'])

    plt.tight_layout()

    plt.savefig('src/data/checkpoint_1.png')
    print('Saved Checkpoint 1 Demo to: ' + 'src/data/checkpoint_1.png')
    
def run_test_demo(dataDir, dataType, annDir, annZipFile, annFile, annURL, capAnnFile):
    '''
    MADE FOR METHODOLOGY 7
    
    Runs our custom cocoapi demo
    takes in all fields from data-params.json found in config folder
    
    returns nothing
    saves demo picture to src/data
    '''
    warnings.filterwarnings("ignore")

    # now check if the annotations have been properly downloaded
    # NOTE: this isn't needed anymore due to the nature of the /teams directory
    # TODO: re-add imports if this func is required again
    #download_annotations(dataDir, dataType, annDir, annZipFile, annFile, annURL)

    print('Initializing COCO api annotations...')
    # initialize COCO api
    coco = COCO(annFile)

    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps = COCO(annFile)

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['cat']);
    imgIds = coco.getImgIds(catIds=catIds);
    print('Finished loading COCO api annotations...')

    # now plot the imagery
    plt.figure(figsize=(20,10))
    columns = 4

    for i in range(len(imgIds[:12])):
        j = coco.loadImgs(imgIds[i])[0]
        I = io.imread(j['coco_url'])
        plt.subplot(3, columns, i + 1)
        plt.imshow(I)

        annIds = coco.getAnnIds(imgIds=j['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        plt.axis('off')

        annIds = coco_caps.getAnnIds(imgIds=j['id']);
        anns = coco_caps.loadAnns(annIds)
        plt.subplot(3,columns,i+1).set_title(anns[0]['caption'])

    plt.tight_layout()

    plt.savefig('test/methodology7_test_demo.png')
    print('Saved Methodology 7 Test Demo to: ' + 'src/data/checkpoint_1.png')