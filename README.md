# tsai-gra-application
Application for a GRA position at GA Tech

This repository contains:

1. Python files for generating TFRecord files from
     - These files should be placed in the following directory structure to execute:
        
        + images
           + test
              - 00000.ppm
              (etc...)
           + train
              - 00000.ppm
              (etc...)
           + test_jpg
           + train_jpg
        - convert_test_to_jpg.py
        - create_train_records.py
        - create_val_records.py
        
2. Config file (faster_rcnn_resnet101_traffic.config)
3. Label map (traffic_sign_label_map.pbtxt)

The record files can be quickly generated using the Python files, or downloaded (along with the final model) from:
