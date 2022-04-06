#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

cd Signboard_Segmentation

conda activate poi

python main.py --input ../image/ --output ../output/output_signboard/

cd ../Scene_Text_Recognition

conda activate str

python demo/merge.py --config-file configs/BAText/VinText/attn_R_50.yaml --input ../image/ --inputfile ../output/output_signboard --output ../output/output_merge --opts MODEL.WEIGHTS trained_model.pth