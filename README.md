# MTCNN_TRAIN
MTCNN_Train Scripts with PyTorch 0.4.0  

## Declaration
**The source code in this repository is mainly from [kuaikuaikim/DFace](https://github.com/kuaikuaikim/DFace).** 
**I reimplemented the part of MTCNN with PyTorch 0.4.0 and made some optimization but most remains unchanged. If you want to know more details, please go to [kuaikuaikim/DFace](https://github.com/kuaikuaikim/DFace)** 


## Introduction



## Questions remains unsolved
1. what's the meaning of annotation_type in imagedb.py/load_annotations()  
2.  


## The Optimizations and Modifications
1. in ./prepare_data/gen_xx_data.py, avoid some unnecessary image data copy operation
2. fix the logical bug of ./tools/image_reader.py, which can't load the last mini_batch when the last minibatch'size is less than the batch_size  
3.  


