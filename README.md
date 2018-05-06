# Face Detection in the Wild

## Objective 
In this work I make an attempt to reimplement the paper " Face Detection with End-to-End Integration of a ConvNet and a 3D Model " ( https://arxiv.org/abs/1606.00850 ) 
This work tries to introduct 3D models into Convolutional Neural Networks by learning the projection parameters from a 3D mean face. 

![alt text](https://github.com/pharish93/FaceDetection/blob/master/Reports/netowrk_sections.png "Netowork Architecture")

In this work is an extension of the Faster RCNN architecture, 
The proposed method addresses two issues in adapting the the Faster RCNN architecture for face detection: 
1. One is to eliminate the heuristic design of predefined anchor boxes in the region proposals network (RPN) by exploiting a 3D mean face model. 
2. The other is to replace the generic RoI (Region-of-Interest) pooling layer with a configuration pooling layer to respect underlying object structures.

Original Implementation of this work can be found at - https://github.com/tfwu/FaceDetection-ConvNet-3D  , Please acknowledge the orginal work if you happen to use this code.

Documentation for this implementation can be found in the Reports folder 

## Prerequisites 
### Environment 

This code in **python 2.7** and is build unsing *Mxnet 1.0 with CUDA 8.0*  . This code uses custom operators, hence it is compulsory to rebuld the framework from sources. 

Please refer to the instruction in https://mxnet.incubator.apache.org/install/index.html for the installation steps. 
Once the build is complete, make ready the python wrappers using these commands 

```
mkdir mxnet
cp -rf incubator-mxnet/python/mxnet/ mxnet
cp -f incubator-mxnet/lib/* mxnet/mxnet
cp -f incubator-mxnet/nnvm/lib/* mxnet/
```
In addition the code opencv 2.4.11 for a few preprocessing and visualization tasks. Please make sure it is present in your enviroment. 

### Data Set

For training, 3D model and key point annotations of images are requied. 

AFLW data set (https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/ ) has been used for experiments in this implementation.
Place the dataset in an appropriate location and specify the path to it in the config.py file. 

This implementation contains Data Loading and a Data Iterator for the AFLW data set.

## Running the Code 

### Training 
End to End training can be done using the file train_end2end.py. 

All the code is present in Face_3D_Models/face_3d folder. The code is organised as follows 
* /core - contains the main functionality for the code , its Data iterator and Training Files 
* /dataset - contains code for reading and loading the AFLW dataset 
* /symbol - has the network used in this implementation 
* /uilts - all visualizaions and other utility functions 

### Demo 
1. Build the mxnet framework from sources using the files in incubator-mxnet
2. Please download the pretrained model - https://drive.google.com/drive/folders/1bwnT6Q2UFRoDEzZYYEo-t3iJaNzD8VvI?usp=sharing 
3. Run the demo.py file 

## Contact Infromation 
If any issue, please contact me : Harish Pullagurla - hpullag@ncsu.edu 
