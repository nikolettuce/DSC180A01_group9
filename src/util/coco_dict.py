from collections import defaultdict
from pycocotools.coco import COCO

def build_dicts(coco_train):
    # get category names
    cats = coco_train.loadCats(coco_train.getCatIds())
    category_dict = defaultdict(str)
    
    for i in cats:
        category_dict[i['id']] = i['name']
    classes = [cat['name'] for cat in cats]
    
    # because the NNCrossEntropy function needs to be index by class from [0,n),
    # we need to rescale out features sequentially from 0 to 79
    new_category_dict = defaultdict(str)
    decoder_category_dict = defaultdict(str) # decode from numbers
    
    counter = 0
    for j in [i[0] for i in category_dict.items()]:
        new_category_dict[counter] = j
        decoder_category_dict[counter] = category_dict[j]
        counter += 1
        
    # encoder dict
    new_category_dict = dict([(value, key) for key, value in new_category_dict.items()])
    
    return new_category_dict, decoder_category_dict