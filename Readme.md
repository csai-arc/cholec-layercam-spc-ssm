# Surgical Phase Classification and Operative Skill Assessment

This repository implements Surgical Phase Classification and Operative Skill Assessment through Spatial Context Aware CNNs and Time-invariant Feature Extracting Autoencoders

### Surgical Phase Classification Training:

This package supports Imagefolder data format. To initiate the training, please use the below command inside 'spc_training' package.

 > python classification_training_layercam.py -a effnetv2_s --data <path to data> --epochs 5 --gpu-id 0 -c <path where checkpoints to be saved> --train-batch 10 --test-batch 1 --weights_load <path to pretrained weights> --optuna_study_db sqlite:///./<path where optuna db to be saved>
 
### Surgical Phase Classification Inference:

This package takes RGB images as input. To initiate the inference, please use the below command inside 'spc_cnn_inference' package.

 > python spr_application_layercam_multilayer.py --cfg cfg/configuration_layercam_multilayer_cam.yaml --mode 1 --eval_mode 1
 

### Surgical Phase Classification Inference using Graph theory:

To initiate the Graph Theory based post processing for surgical phase classification, please execute 'spc_post_processing.ipynb' notebook inside 'spc_graph_inference' package.


### Surgical Skill Measurement:

To initiate the SSM measure, please execute 'ssm_inference_2d_cpd_ds_ws10.ipynb' notebook inside 'ssm_training_inference' package.



For trained network models please contact the author.
