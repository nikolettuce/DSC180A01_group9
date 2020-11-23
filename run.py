#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')
sys.path.insert(0, 'src/util')

from src.util import download_annotations
from src.util import run_demo
from src.model import model
from src.util import coco_dataloader
from src.util import coco_dict

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import torch
from collections import defaultdict

from pycocotools.coco import COCO


def main(targets):
    '''
    runs basic demo for COCO api.
    
    configure filepaths based on data-params.json, however all should work by default.
    '''
    if 'data' in targets:
        with open('config/data-params.json') as fh:
                data_cfg = json.load(fh)
                print(data_cfg['dataDirJSON'])

        # load paths
        dataDirJSON = data_cfg['dataDirJSON']
        dataType = data_cfg['dataTypeVal']
        annDir = data_cfg['annDir'].format(dataDirJSON)
        annZipFile = data_cfg['annZipFile'].format(dataDirJSON, dataType)
        annFile = data_cfg['annFile'].format(annDir, dataType)
        annURL = data_cfg['annURL'].format(dataType)
        capAnnFile = data_cfg['capAnnFile'].format(dataDirJSON, dataType)

        # run coco_demo.py function run_demo
        run_demo(dataDirJSON, dataType, annDir, annZipFile, annFile, annURL, capAnnFile)
        
    elif 'test_meth' in targets:
        with open('test/test-data-params.json') as fh:
                data_cfg = json.load(fh)
                print(data_cfg['dataDirJSON'])

        # load paths
        dataDirJSON = data_cfg['dataDirJSON']
        dataType = data_cfg['dataTypeVal']
        annDir = data_cfg['annDir'].format(dataDirJSON)
        annZipFile = data_cfg['annZipFile'].format(dataDirJSON, dataType)
        annFile = data_cfg['annFile'].format(annDir, dataType)
        annURL = data_cfg['annURL'].format(dataType)
        capAnnFile = data_cfg['capAnnFile'].format(dataDirJSON, dataType)

        # run coco_demo.py function run_demo
        run_test_demo(dataDirJSON, dataType, annDir, annZipFile, annFile, annURL, capAnnFile)
        
    elif 'train' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        # build paths
        TRAIN_PATH = data_cfg['dataDir'] + '/' + data_cfg['dataTypeTrain']
        VAL_PATH = data_cfg['dataDir'] + '/' + data_cfg['dataTypeVal']
        JSON_TRAIN_PATH = data_cfg['annFile'].format(data_cfg['dataDir'] + '/anno2017', data_cfg['dataTypeTrain'])
        JSON_VAL_PATH = data_cfg['annFile'].format(data_cfg['dataDir'] + '/anno2017', data_cfg['dataTypeVal'])
        
        # initiate coco for dict
        coco = COCO(JSON_TRAIN_PATH)
        
        # build encoder/decoder dicts 
        encoder_dict, decoder_dict = coco_dict.build_dicts(coco)
        
        # change to gpu 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("COMPUTING AS: {}".format(device))
        
        # transforms 
        transform = transforms.Compose([ 
                transforms.Resize([240,240]),
                transforms.ToTensor()
        ])
        
        # define CNN
        net = model.Net()
        net = net.to(device) # to gpu
        
        # initiate custom torchvision dset
        coco_train = coco_dataloader.CocoDetection(root = TRAIN_PATH, annFile = JSON_TRAIN_PATH, transform = transform )
        
        # dataloaders
        trainloader = torch.utils.data.DataLoader(coco_train, batch_size=5,
                                       shuffle=True, num_workers=2)

        # run the model
        model.run_model(
            loader = trainloader,
            net = net,
            epochs = model_cfg['epochs'],
            lr = model_cfg['lr'],
            momentum = model_cfg['momentum'],
            modelDir = model_cfg['modelDir'],
            device = device
        )
        
    elif 'test' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        # load the modelDir
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
        
        # build paths
        TRAIN_PATH = data_cfg['dataDir'] + '/' + data_cfg['dataTypeTrain']
        VAL_PATH = data_cfg['dataDir'] + '/' + data_cfg['dataTypeVal']
        JSON_TRAIN_PATH = data_cfg['annFile'].format(data_cfg['dataDir'] + '/anno2017', data_cfg['dataTypeTrain'])
        JSON_VAL_PATH = data_cfg['annFile'].format(data_cfg['dataDir'] + '/anno2017', data_cfg['dataTypeVal'])
        
        # transforms 
        transform = transforms.Compose([ 
                transforms.Resize([240,240]),
                transforms.ToTensor()
        ])
        
        # load test data
        coco_test =  coco_dataloader.CocoDetection(root = VAL_PATH, annFile = JSON_VAL_PATH, transform = transform )
        testloader = torch.utils.data.DataLoader(coco_test, batch_size=5,
                               shuffle=True, num_workers=2)
        
        # change to gpu 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("COMPUTING AS: {}".format(device))
        
        net = model.Net()
        net = net.to(device)
        
        print('Loading model from {}...'.format(model_cfg['modelDir']))
        net.load_state_dict(torch.load(model_cfg['modelDir']))
        
        model.test_model(net, testloader, device)
        
        
            
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)