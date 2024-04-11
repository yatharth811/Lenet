<!-- All the inputs in Subtask 1 and 2 are taken using command line input. -->
<!-- Input format:

N : size of input matrix
N x N input matrix
f : size of kernel
f x f kernel matrix
pad: size of padding in one side
tanh/relu : activation function
max/avg : pooling
filter_size: pooling filter size
sigmoid/softmax

Output:
Softmax/Sigmoid 
-->

<!-- Subtask 1 -->
nvcc src/assignment2_subtask1.cu -o main.out
./main.out


<!-- Subtask 2 -->
nvcc src/assignment2_subtask2.cu -o main.out
./main.out


<!-- Prepocess all the images and stores processed images in pre-proc-img directory -->
<!-- It also generates img_path.txt which contain paths of all the processed images. This file is used in Subtask4-->
<!-- Run this before running lenet architecture-->
python3 preprocess.py

<!-- Subtask 3 -->
nvcc src/assignment2_subtask3.cu -o lenet.out
./lenet.out path_to_img
<!-- Example - ./lenet.out img/000000-num7.png-->

<!-- Subtask 4 -->
nvcc src/assignment2_subtask4.cu -o lenet.out
./lenet.out img_path.txt
