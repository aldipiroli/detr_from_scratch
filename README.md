# DETR from scratch 
Implementing from scratch the paper "[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)" ECCV 2020.

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/detr_from_scratch
pip install -r requirements.txt
``` 
### Train 
``` 
python train.py detr/config/config.yaml
``` 

### Evaluate 
``` 
python evaluate.py detr/config/config.yaml --ckpt path/to/ckpt # or --ckpt_folder path/to/ckpt_folder
``` 

### Results
Detection results on the VOC2012 validation set based on the traing epoch. 
- Top: raw output of the model. 
- Bottom: output postprocessed with nms + attention maps for each query.

![](assets/teaser.gif)