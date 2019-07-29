# Waste management using Deep Learning in Raspberry Pi 3
## Overview
The Raspberry pi automatically detects and identifies wastes through pi camera which coonected to 3 waste bins (e waste ,organic,recycle).
System identify wastes types and triggers a bin to open. 



 ## Requirements
 1. Tensorflow
 2. python 3
 3. Dependencies for Tensorflow
 4. OpenCV
 
 ## Getting Started
* Clone the tensorflow official repository from https://github.com/tensorflow/models.git
* I Have used SSD mobilenet model for training since the  detection happens to be in  raspberry pi , which has considerably low amount of specification

## Labelling the image
Use LabelIMG tool

## Generate Training Data
`python xml_to_csv.py`
This coomand will convert all xml files of labelled datasets to csv file 

## Generate TF recordds
`python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record`
`python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record`

### Edit Labelmap.pbtxt file since i have 3 classes 

## Train the model
`python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config`

*trained in Desktop computer with Nvidia GTX 1070*
 
## Testing the model in raspberry pi with picamera
 Exporting Inference graph from Desktop `python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph`
 
 Place inference graph folder in raspberry pi object detection folder
 
 Test it using `python 3 object_detection picamera.py`
 
 
 ## References
 https://github.com/tensorflow/models
 
 https://github.com/EdjeElectronics
