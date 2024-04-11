#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <cassert>
#include <chrono>
#include "utils.hpp"

using namespace std::chrono;

typedef float float1D;
typedef float float2D;
typedef float float3D;
typedef float float4D;

// Weights and Biases
float3D conv1_wts[500];
float1D conv1_bias[20];
float4D conv2_wts[25000];
float1D conv2_bias[50];
float4D fclayer1_wts[400000];
float1D fclayer1_bias[500];
float4D fclayer2_wts[5000];
float1D fclayer2_bias[10];

// Number of Streams
#define NUM_STREAMS 32
cudaStream_t stream[NUM_STREAMS];

// Gpu constant memory for weights
__constant__ float2D kernel_2D[500];

__global__ void conv2DKernel_shared(float2D *input, float2D *conv, int n, int f, float bias, int filter_num){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Shared Memory Optimization
  __shared__ float N_ds[28][28];
  if(row<n && col<n){
    N_ds[row][col] = input[row*n+col];
  }
  __syncthreads();

  if (row < n - f + 1 and col < n - f + 1) {
    float temp = bias;
    for (int k = 0; k < f; k += 1) {
      for (int l = 0; l < f; l += 1) {
        assert(filter_num*25+k*f+l<500);
        temp += kernel_2D[filter_num*25+k*f+l] * N_ds[k+row][l+col];
      }
    }
    conv[row * (n - f + 1) + col] = temp;
  }
  
}

__global__ void conv3DKernel_shared(float3D *input, float3D *kernel, float2D *conv, int n, int f, int channels, float bias) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Shared Memory Optimization
  __shared__ float N_ds[12][12][20];

  if(row<n && col<n){
    for (int channel = 0; channel < channels; channel += 1) {
      N_ds[row][col][channel] = input[channel * n * n +  row * n + col];
    }
  }
  __syncthreads();

  if (row < n - f + 1 and col < n - f + 1) {
    float temp = 0;
    for (int k = 0; k < f; k += 1) {
      for (int l = 0; l < f; l += 1) {
        for (int channel = 0; channel < channels; channel += 1) {
          temp += kernel[channel * f * f + k * f + l]*N_ds[k+row][l+col][channel];
        }
      }
    }
    conv[row * (n - f + 1) + col] = temp + bias;
  }
}

__global__ void max_pool_kernel(float *input, float *output, int n, int f){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int out_height = n/f;
  int out_width = n/f;
  if(row<out_height && col<out_width){
    float temp = -1e9;
    int st_row = row*f, st_col = col*f;
    for(int i=0; i<f; i++){
        for(int j=0; j<f; j++){
            temp = max(temp,input[(st_row+i)*n+(st_col+j)]);
        }
    }
    output[row*out_width+col] = temp;
  }
}

__global__ void fclayer1_kernel(float3D *input, float3D *output, float4D* weights, float *bias){
  int row = threadIdx.x;

  // Shared Memory Optimization
  __shared__ float N_ds[800];

  if(row<800){
    N_ds[row] = input[row];
  }

  __syncthreads();

  if(row<500){
    float temp = bias[row];
    for (int j = 0; j < 50; j += 1) {
      for (int k = 0; k < 4; k += 1) {
        for (int l = 0; l < 4; l += 1) {
          temp += (N_ds[j * 16 + k * 4 + l] * weights[row * 800 + j * 16 + k * 4 + l]);
        }
      }
    }
    output[row] = (temp > 0 ? temp : 0);
  }
}

__global__ void fclayer2_kernel(float3D *input, float3D *output, float4D* weights, float *bias){
  int row = threadIdx.x;
  // Shared Memory Optimization
  __shared__ float N_ds[500];

  if(row<500){
    N_ds[row] = input[row];
  }

  __syncthreads();

  if(row<10){
    float temp = bias[row];
    for (int j = 0; j < 500; j += 1) {
      temp += (N_ds[j] * weights[row * 500 + j]);
    }
    output[row] = temp;
  }
}

void conv2D(float2D *input, float2D *conv, int n, int f, float bias, int filter_num, int stream_id) {

  // Dim
  dim3 block_dim(32, 32);
  
  conv2DKernel_shared<<<1, block_dim, 0, stream[stream_id]>>>(input, conv, n, f, bias, filter_num);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
}



void conv3D(float3D *input, float3D* kernel, float2D *conv, int n, int f, int channels, float bias, int stream_id) {

  // Dim
  dim3 block_dim(32, 32);

  // Call the kernel
  conv3DKernel_shared<<<1, block_dim, 0, stream[stream_id]>>>(input, kernel, conv, n, f, channels, bias);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));

}


// LeNet 5 Architecture
// 28 * 28 image input (*) 20 5x5 filters + 20 * bias
// input: float[28][28]
// filters: float[20][5][5]
// output: float[20][24][24]
void conv1(float2D *input, float3D *conv, float1D* bias, int stream_id) {
  for (int i = 0; i < 20; i += 1) {
    conv2D(input, conv + i * 24 * 24, 28, 5, bias[i], i, stream_id);
  }
}

// Max pooling here
// 20 input 24 x 24 and 2x2 kernel -> 20 output 12x12
// input: float[20][24][24]
// output: float[20][12][12]
void pool1(float3D *input, float3D *output, int stream_id) {
  dim3 block_dim(32, 32);
  for (int i = 0; i < 20; i += 1) {
    max_pool_kernel<<<1,block_dim,0,stream[stream_id]>>>(input + i * 24 * 24, output + i * 12 * 12, 24, 2);
  }
}


// 2nd convolutional layer
// input: float[20][12][12]
// filters: float[50][20][5][5]
// output: float[50][8][8]
void conv2(float3D *input, float3D *output, float4D *weights, float *bias, int stream_id) {
  for (int i = 0; i < 50; i += 1) {
    conv3D(input, weights+i*500, output+i*64, 12, 5, 20, bias[i], stream_id);
  }
}

// Max pooling here
// input: float[50][8][8]
// output: float[50][4][4]
void pool2(float3D *input, float3D *output, int stream_id) {
  dim3 block_dim(32, 32);
  for (int i = 0; i < 50; i += 1) {
    max_pool_kernel<<<1, block_dim, 0, stream[stream_id]>>>(input + i * 64, output + i * 16, 8, 2);
  }
}

// FCLayer 1 + ReLu
// input: float[50][4][4]
// filters: float[500][50][4][4]
// output: float[500][1][1]
void fclayer1(float3D *input, float3D *output, float4D* weights, float *bias, int stream_id) {
  fclayer1_kernel<<<1, 1024, 0, stream[stream_id]>>>(input, output, weights, bias);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));

}

// FCLayer 2 + ReLu
// input: float[500][1][1]
// filters: float[10][500][1][1]
// output: float[10][1][1]
void fclayer2(float3D *input, float3D *output, float4D *weights, float *bias, int stream_id) {
  fclayer2_kernel<<<1, 512, 0, stream[stream_id]>>>(input, output, weights, bias);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
}

// Output layer
// input: float[10][1][1]
int outputLayer(float3D* input) {
  int classLabel{-1};
  float prob = 0.0F;
  for (int i = 0; i < 10; i += 1) {
    if (input[i] > prob) {
      prob = input[i];
      classLabel = i;
    }
  }
  return classLabel;
}

int main(int argc, char** argv) {

  if(argc!=2){
    std::cerr << "Wrong Arguments Passed, pass 0 (stream off) or 1 (stream on)\n";
    return 0;
  }

  int is_stream = std::stoi(argv[1]);
  // Images file name
  std::string filename = "img_path.txt";
  auto start = high_resolution_clock::now();

  // load weights and biases.

  if (!load_weights(conv1_wts, conv1_bias, 500, 20, "weights/conv1.txt")) {
    std::cout << "Unable to load conv1 weights." << std::endl;
    return 0;
  }
  cudaMemcpyToSymbol(kernel_2D, conv1_wts, 500 * sizeof(float),0,cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error1: %s\n", cudaGetErrorString(err));
  
  if (!load_weights(conv2_wts, conv2_bias, 25000, 50, "weights/conv2.txt")) {
    std::cout << "Unable to load conv2 weights." << std::endl;
    return 0;
  }
  
  if (!load_weights(fclayer1_wts, fclayer1_bias, 400000, 500, "weights/fc1.txt")) {
    std::cout << "Unable to load fclayer1 weights." << std::endl;
    return 0;
  }
  
  if (!load_weights(fclayer2_wts, fclayer2_bias, 5000, 10, "weights/fc2.txt")) {
    std::cout << "Unable to load fclayer2 weights." << std::endl;
    return 0;
  }

  // Allocate Pinned Memory
  float2D *image;
  cudaMallocHost((void**)&image, 784*NUM_STREAMS*sizeof(float));

  float3D *fclayer2_output;
  cudaMallocHost((void**)&fclayer2_output, 10*NUM_STREAMS*sizeof(float));
 
  // Allocate Device Memory
  float3D *d_conv1_output;
  cudaMalloc(&d_conv1_output,11520*NUM_STREAMS*sizeof(float));

  float3D *d_pool1_output;
  cudaMalloc(&d_pool1_output,2880*NUM_STREAMS*sizeof(float));

  float4D *d_conv2_wts;
  cudaMalloc(&d_conv2_wts,25000*sizeof(float));
  cudaMemcpy(d_conv2_wts,conv2_wts,25000*sizeof(float),cudaMemcpyHostToDevice);

  float3D *d_conv2_output;
  cudaMalloc(&d_conv2_output,3200*NUM_STREAMS*sizeof(float));

  float3D *d_pool2_output;
  cudaMalloc(&d_pool2_output,800*NUM_STREAMS*sizeof(float));

  float3D *d_fclayer1_output;
  cudaMalloc(&d_fclayer1_output,500*NUM_STREAMS*sizeof(float));

  float3D *d_fclayer1_wts;
  cudaMalloc(&d_fclayer1_wts,400000*sizeof(float));
  cudaMemcpy(d_fclayer1_wts,fclayer1_wts,400000*sizeof(float),cudaMemcpyHostToDevice);

  float3D *d_fclayer1_bias;
  cudaMalloc(&d_fclayer1_bias,500*sizeof(float));
  cudaMemcpy(d_fclayer1_bias,fclayer1_bias,500*sizeof(float),cudaMemcpyHostToDevice);

  float3D *d_fclayer2_output;
  cudaMalloc(&d_fclayer2_output,10*NUM_STREAMS*sizeof(float));

  float3D *d_fclayer2_wts;
  cudaMalloc(&d_fclayer2_wts,5000*sizeof(float));
  cudaMemcpy(d_fclayer2_wts,fclayer2_wts,5000*sizeof(float),cudaMemcpyHostToDevice);

  float3D *d_fclayer2_bias;
  cudaMalloc(&d_fclayer2_bias,10*sizeof(float));
  cudaMemcpy(d_fclayer2_bias,fclayer2_bias,10*sizeof(float),cudaMemcpyHostToDevice);

  float2D *d_image;
  cudaMalloc(&d_image,784*NUM_STREAMS*sizeof(float));

  // Load image names and their labels
  std::vector<std::string> images = load_image_paths(filename);
  std::vector<std::vector<float>> images_prob;

  // Initialize cuda streams
  for (int i = 0; i < NUM_STREAMS; ++i)
    cudaStreamCreate(&stream[i]);

  if(is_stream){

    int num_itr = (images.size()+NUM_STREAMS-1)/NUM_STREAMS;
    for(int i=0; i<num_itr; i++){
      int st = i*NUM_STREAMS, en = std::min((i+1)*NUM_STREAMS,(int)images.size());
      // Distributing images to streams
      for(int j=st; j<en; j++){
          int idx = j%NUM_STREAMS;
          if (!load_image(image+idx*784, 784,"pre-proc-img/"+images[j])) {
              std::cout << "Unable to load image " << "pre-proc-img/"+images[j] << std::endl;
              return 0;
          }
          // CNN Architecture
          cudaMemcpyAsync(d_image+idx*784,image+idx*784,784*sizeof(float),cudaMemcpyHostToDevice,stream[idx]);
          conv1(d_image+idx*784, d_conv1_output+idx*11520, conv1_bias, idx);
          pool1(d_conv1_output+idx*11520, d_pool1_output+idx*2880, idx);
          conv2(d_pool1_output+idx*2880, d_conv2_output+idx*3200, d_conv2_wts, conv2_bias,idx);
          pool2(d_conv2_output+idx*3200, d_pool2_output+idx*800, idx);
          fclayer1(d_pool2_output+idx*800, d_fclayer1_output+idx*500, d_fclayer1_wts, d_fclayer1_bias, idx);
          fclayer2(d_fclayer1_output+idx*500, d_fclayer2_output+idx*10, d_fclayer2_wts, d_fclayer2_bias, idx);
          cudaMemcpyAsync(fclayer2_output+idx*10, d_fclayer2_output+idx*10, 10*sizeof(float), cudaMemcpyDeviceToHost, stream[idx]);
      }
      // Wait for computation to finish in all streams
      cudaDeviceSynchronize();
      // Generate prediction
      for(int j=st; j<en; j++){
        int idx = j%NUM_STREAMS;
        std::vector<float> prob = top5_prob(fclayer2_output+idx*10,10);
        images_prob.push_back(prob);
      }
    }
  }
  else{
    
    for(int i=0; i<images.size(); i++){
      if (!load_image(image, 784,"pre-proc-img/"+images[i])) {
          std::cout << "Unable to load image " << "pre-proc-img/"+images[i] << std::endl;
          return 0;
      }
      // CNN Architecture
      cudaMemcpyAsync(d_image,image,784*sizeof(float),cudaMemcpyHostToDevice,stream[0]);
      conv1(d_image, d_conv1_output, conv1_bias, 0);
      pool1(d_conv1_output, d_pool1_output, 0);
      conv2(d_pool1_output, d_conv2_output, d_conv2_wts, conv2_bias, 0);
      pool2(d_conv2_output, d_pool2_output, 0);
      fclayer1(d_pool2_output, d_fclayer1_output, d_fclayer1_wts, d_fclayer1_bias, 0);
      fclayer2(d_fclayer1_output, d_fclayer2_output, d_fclayer2_wts, d_fclayer2_bias, 0);
      cudaMemcpyAsync(fclayer2_output, d_fclayer2_output, 10*sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
      cudaDeviceSynchronize();
      std::vector<float> prob = top5_prob(fclayer2_output,10);
      images_prob.push_back(prob);
    }
  }

  for(int i=0; i<images.size(); i++) writeToFile(images_prob[i],"output/"+images[i]+".txt");

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time taken to process 10000 images in ms = " << duration.count() << "\n";
  
  // Free pinned memory and device memory 
  cudaFreeHost(image);
  cudaFreeHost(fclayer2_output);
  cudaFree(d_image);
  cudaFree(d_conv1_output);
  cudaFree(d_conv2_output);
  cudaFree(d_pool1_output);
  cudaFree(d_pool2_output);
  cudaFree(d_fclayer1_output);
  cudaFree(d_fclayer1_wts);
  cudaFree(d_fclayer1_bias);
  cudaFree(d_fclayer2_output);
  cudaFree(d_fclayer2_wts);
  cudaFree(d_fclayer2_bias);

  return 0;
}