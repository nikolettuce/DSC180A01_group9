
- dataDir, the directory of your cocoapi folder from https://github.com/philferriere/cocoapi
- dataType, the validation or test COCO data from either 2017 or 2015, see COCO website for all options
- annDir, directory in which your annotations are saved in the format {}/annotations, where {} will be formatted with dataDir.
- annZipFile, name of zipfile downloaded from annURL, formatted {}/annotations_train{}.zip, where {}0 will be replaced with dataDir, and {}1 will be replaced with dataType
- annFile, fname of annotations file, same rules as annZipFile
- annURL, URL from which the annotations will be downloaded from the COCO website or other source.
- capAnnFile, fname of captions annotations file, same rules as annZipFile


The default parameters for this project are:
{
    "dataDir": "src/data/cocoapi",
    "dataType": "val2017",
    "annDir": "{}/annotations",
    "annZipFile": "{}/annotations_train{}.zip",
    "annFile": "{}/instances_{}.json",
    "annURL": "http://images.cocodataset.org/annotations/annotations_train{}.zip",
    "capAnnFile":"{}/annotations/captions_{}.json"
}