#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda create -n poi python=3.6

conda activate poi

cd Signboard_Segmentation

pip install -r requirements.txt

conda create -n str python=3.7

conda activate str

cd ../Scene_Text_Recognition

pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 torchtext==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance dict-trie mlflow

python -m pip install detectron2==0.2.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html

python setup.py install