# DSC180A01 Explainable AI
By: Nikolas Racelis-Russell A15193325, Sohyun Lee A15139672
### Introduction
 When running an image classification model, users run through the problem of not understanding why they should trust their model. This project will investigate how to make AI more understandable by giving users options by including additional knowledge from our models and presenting it to the user. 
 The facet through which this concept is presented is object recognition, using the COCO, Common Objects in Context, dataset and COCO python api to interact with said dataset. To make AI more unstandable to users, the machine can explain its result and offer additional knowledge - by offering succinct insight into the model’s intricacies to allow for more nuanced tuning. If a user’s model is making a mistake constantly, having methods to diagnose why the model is making that mistake (i.e. visualization of key points such as highlighting humans or objects in an image, or the mistaken highlighting thereof), users will have an easier time understanding why their model isn’t working as planned and can act accordingly.
### Data
 Image data with annotations, called COCO datasets, will be used for this project. The annotation in coco data highlights key points in the image, such as a human or a cat, and so on. The COCO dataset was chosen because it contains built in object segmentation, recognition in context, superpixel stuff segmentation, 330,000 images with other 200,000 of them already being labeled, 1.5 million object instances, 80 categories of objects, 91 stuff categories, at least 5 captions per image, and 250,000 people objects with key points (annotations)
### Run.py 
 Please clone this repo onto a UCSD dsmlp cluster, as the dataset is stored in the public datasets folder under COCO-2017 and COCO-2015. However, annotations and such will be check for and downloaded according to our run.py
Additionally, please run the command: pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI 

The code here uses the COCO api to show 12 images of cats, their highlighted figures (or what the model "perceives" as cats), and a caption for the image.
 
### Troubleshooting
 If the submodule under src/data/cocoapi isn’t properly downloaded. Download it manually from https://github.com/philferriere/cocoapi and follow the instructions given there

### Responsibilities
- Nikolas Racelis-Russell worked on coding the run.py and wrote the introduction section
- Sohyun Lee worked on coding the run.py and wrote the data section
