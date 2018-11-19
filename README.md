# tsai-gra-application
Application for a GRA position at GA Tech

This repository contains:

1. Python files for generating TFRecord files from .ppm image files
     
     These files should be placed in the following directory structure to execute:   

       (+) images/
           (+) test/
              (-) 00000.ppm
              (etc...)
           (+) train/
              (-) 00000.ppm
              (etc...)
           (+) test_jpg/
           (+) train_jpg/

       (+) data/
        
       (-) convert_test_to_jpg.py
       (-) create_train_records.py
       (-) create_val_records.py
        
2. Config file (faster_rcnn_resnet101_traffic.config)
3. Label map (traffic_sign_label_map.pbtxt)

The resulting TFRecord files, as well as the final trained model weights (saved as "frozen_inference_graph.pb"), can be downloaded at:
https://www.dropbox.com/home/tsai-gra-application
