import os, sys, zipfile
import urllib.request

def download_annotations(dataDir, dataType, annDir, annZipFile, annFile, annURL):
    '''
    download annotations from coco website
    '''
    print("Checking for annotations in " + annDir)
    
    # Download data if not available locally
    if not os.path.exists(annDir):
        os.makedirs(annDir)
    if not os.path.exists(annFile):
        if not os.path.exists(annZipFile):
            print ("Downloading zipped annotations to " + annZipFile + " ...")
            with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print ("... done downloading.")
        print ("Unzipping " + annZipFile)
        with zipfile.ZipFile(annZipFile,"r") as zip_ref:
            zip_ref.extractall(dataDir)
        print ("... done unzipping")
    print ("Will use annotations in " + annFile)
