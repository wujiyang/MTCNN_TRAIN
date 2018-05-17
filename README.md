# MTCNN_TRAIN
MTCNN_Train Scripts with PyTorch 0.4.0  

## Declaration
**The source code in this repository is mainly from [kuaikuaikim/DFace](https://github.com/kuaikuaikim/DFace).** 
**I reimplemented the part of MTCNN with PyTorch 0.4.0 and made some optimizations but most remains unchanged. If you want to know more details, please go to [kuaikuaikim/DFace](https://github.com/kuaikuaikim/DFace)** 

---
## Introduction 

This project is still in progess, I will finish it in my spare time as soon as possible !

## The Optimizations 
1. avoid some unnecessary image data copy operation in ./prepare_data/gen_xx_data.py.
2. when generating the training imglist, split some for validation set.
3. fix the bug that data_loader can't load the last mini_batch when the last minibatch'size is less than the batch_size in ./tools/image_reader.py, 
4. add validation process in model training.





