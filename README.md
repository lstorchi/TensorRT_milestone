# TensorRT_milestone
Sample code for TensorRT inference on CMS DL models.<br />

Trk_Doublets.cpp --> main code for inference on doublets with TRT <br />
trt_env.sh --> sample environment to use TensorRT <br />
compile.txt --> info to compile .cpp code <br />
pixel_only.pb --> frozen model that only uses pixels (no features added, no batch normalization) <br />
pixel_and_features.pb --> complete frozen model (with features added and batch normalization) <br />
pixel_and_features_noBatchNorm.pb --> frozen model that uses features but not batch normalization <br />

pixel_model.py --> script that uses Tensorflow 2.1 for the model creation <br />
Trt_builder.py --> model made with pixel_model.py is transformed into a TensorRT one <br />
TRT_inference.py --> run inference with the model made by Trt_builder.py <br />

NB: The three .py script uses Cuda 10.1 and TensorRT 6.0 for compatibility reasons

