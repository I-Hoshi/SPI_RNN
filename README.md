SIngle-pixel imaging using a recurrent neural network combined with convolutional layers
====

## Description
The source code used for single-pixel imaging using a recurrent neural network is available in this repository.  
The paper in [1] is the basis for the entire network.

DCAN_RNN.py : first step learning  
~_predict.py : for inference  
~test.py : for test  
~pattern_binarized_RNN.py : second step learning for binarization  
RNNGI.py : single-pixel imaging using a recurrent neural network without pattern learning  

These codes can probably work without change.  
If they can't work, please modify them properly to work.   



## Requirement
Please cite the following reference in papers using these source codes:  
`I. Hoshi, T. Shimobaba, T. Kakue, T. Ito, "Single-pixel imaging using a recurrent neural network combined with convolutional layers," Opt. Express 28 (23), 34069 (2020).`  

The following environment is recommended.  
python 3.5.4  
cuda 9.0  
cudnn 7.1.4  
keras 2.2.4  
tensorflow-gpu 1.8.0  

## Licence

[MIT](https://github.com/I-Hoshi/SPI_RNN/blob/master/LICENSE)

## Reference
[1] C. F. Higham, R. M. Smith, M. J. Padgett, and M. P. Edgar, “Deep Learning for Real-Time Single-Pixel Video,” Sci. Rep. 8 (1), 2369 (2018).

## Other repositories
[I-Hoshi](https://github.com/I-Hoshi)
