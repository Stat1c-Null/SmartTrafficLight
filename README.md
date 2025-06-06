# SmartTrafficLight
Smart Traffic Light System using Computer Vision

## Installation
### Create Miniconda environment 
conda create --name smart_traffic_light python=3.11

### Installed required packages
conda env create -f environment.yaml

### Training model
If you want to train your own version, download coco2017 dataset from: https://cocodataset.org/#download

### Download model and pre-trained weights
https://drive.google.com/drive/folders/1ULYuaMoGNzZbDCfYbzHliy1DFQelQuFy?usp=sharing

Place pre-trained weights into main repo folder

Place model into output folder

### Clone detectron2 repo
git clone https://github.com/facebookresearch/detectron2/tree/main