# Assignment 3

Note: Please change “data_path” based on your own data directory.  

## Usage

Train for t2w data:
```
python train_eval.py \
    --learning_ratelr 1e-5 \
    --BatchSize 8 \
    --epochs 50 \
    --data 'T2w' \
    --loss 'dice_loss' \
    --data_path 'D:/Courses/CISC_881_Medical_Imaging/PICAI_dataset/' \
    --momentum 0.9 \
    --weight-decay 2e-4 \
    --save_step 10 \
    --min_delta 0.07 \

Train for adc data:
```
python train_eval.py \
    --learning_ratelr 1e-5 \
    --BatchSize 8 \
    --epochs 50 \
    --data 'Adc' \
    --loss 'dice_loss' \
    --data_path 'D:/Courses/CISC_881_Medical_Imaging/PICAI_dataset/' \
    --momentum 0.9 \
    --weight-decay 2e-4 \
    --save_step 10 \
    --min_delta 0.07 \

Train for hbv data:
```
python train_eval.py \
    --learning_ratelr 1e-5 \
    --BatchSize 8 \
    --epochs 50 \
    --data 'Hbv' \
    --loss 'dice_loss' \
    --data_path 'D:/Courses/CISC_881_Medical_Imaging/PICAI_dataset/' \
    --momentum 0.9 \
    --weight-decay 2e-4 \
    --save_step 10 \
    --min_delta 0.07 \

----
For visualization
```
python Visual_result.py 
---------

The directory tree structure for the Dataset folder must be like this (containing images):


├── PICAI_dataset
│   ├── picai_labels-main
│   │   ├──anatomical_delineations
│   │   │  ├──whole_gland
│   │   │     ├──AI
│   │   │        ├──Bosma22b
│   │   │
│   │   ├──csPCa_lesion_delineations
│   │   │  ├──AI
│   │   │     ├──Bosma22a
│   │   └──clinical_information
│   │
│   ├── picai_public_images_fold0
│   ├── picai_public_images_fold1
│   ├── picai_public_images_fold2
│   ├── picai_public_images_fold3
│   └── picai_public_images_fold4

## Requirements
- python 3.9+
- torch 1.10.1+
- torchvision 0.11.2+
- numpy 1.22.0+
- SimpleITK 2.2.1+
