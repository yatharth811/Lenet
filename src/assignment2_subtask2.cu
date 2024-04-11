#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>


__global__ void conv2DKernel(float *input, float *kernel, float *conv, int n, int f) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n - f + 1 and col < n - f + 1) {
    float temp = 0;
    for (int k = 0; k < f; k += 1) {
      for (int l = 0; l < f; l += 1) {
        temp += kernel[k * f + l] * input[(k + row) * n + (l + col)];
      }
    }
    conv[row * (n - f + 1) + col] = temp;
  }
}


__global__ void max_pool_kernel(float *input, float *output, int n, int f){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n - f + 1 and col < n - f + 1) {
    float temp = -1e9;
    for (int k = 0; k < f; k += 1) {
      for (int l = 0; l < f; l += 1) {
        temp = max(temp, input[(k + row) * n + (l + col)]);
      }
    }
    output[row * (n - f + 1) + col] = temp;
  }
}

__global__ void avg_pool_kernel(float *input, float *output, int n, int f){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n - f + 1 and col < n - f + 1) {
    float temp = 0.0F;
    for (int k = 0; k < f; k += 1) {
      for (int l = 0; l < f; l += 1) {
        temp += input[(k + row) * n + (l + col)];
      }
    }
    output[row * (n - f + 1) + col] = temp / (f * f);
  }

}



void conv2D(float *input, float* kernel, float *conv, int n, int f, int pad) {

  float* ninput = new float[(n + 2 * pad) * (n + 2 * pad)];
  memset(ninput, 0, sizeof(ninput));

  for (int i = 0; i < n; i += 1) {
    for (int j = 0; j < n; j += 1) {
      ninput[(i + pad) * (n + 2 * pad) + (j + pad)] = input[i * n + j]; 
    }
  }

  // Update with padding
  n += 2 * pad;

  // Set up data on device
  float *d_input;
  float *d_conv;
  float *d_kernel;

  // Allocate device memory
  cudaMalloc(&d_input, n * n * sizeof(float));
  cudaMalloc(&d_kernel, f * f * sizeof(float));
  cudaMalloc(&d_conv, (n - f + 1) * (n - f + 1) * sizeof(float));

  // Copy data to the device
  cudaMemcpy(d_input, ninput, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, f * f * sizeof(float), cudaMemcpyHostToDevice);

  // Dim - Adjust according to need
  dim3 block_dim(16, 16);

  // Call the kernel
  conv2DKernel<<<1, block_dim>>>(d_input, d_kernel, d_conv, n, f);

  // Copy the result back to the CPU
  cudaMemcpy(conv, d_conv, (n - f + 1) * (n - f + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  // Synchronize here
  cudaDeviceSynchronize();

  // Free Device Memory
  cudaFree(d_input);
  cudaFree(d_conv);
  cudaFree(d_kernel);

  // Free host memory
  delete[] ninput;
}


void max_pool(float *input, float* output, int n, int f) {

  // Set up data on device
  float *d_input;
  float *d_output;

  // Allocate device memory
  cudaMalloc(&d_input, n * n * sizeof(float));
  cudaMalloc(&d_output, (n - f + 1) * (n - f + 1) * sizeof(float));

  // Copy data to the device
  cudaMemcpy(d_input, input, n * n * sizeof(float), cudaMemcpyHostToDevice);

  // Dim - Adjust according to need
  dim3 block_dim(16, 16);

  // Call the kernel
  max_pool_kernel<<<1, block_dim>>>(d_input, d_output, n, f);

  // Copy the result back to the CPU
  cudaMemcpy(output, d_output, (n - f + 1) * (n - f + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  // Synchronize here
  cudaDeviceSynchronize();

  // Free Device Memory
  cudaFree(d_input);
  cudaFree(d_output);

}


void avg_pool(float *input, float* output, int n, int f) {

  // Set up data on device
  float *d_input;
  float *d_output;

  // Allocate device memory
  cudaMalloc(&d_input, n * n * sizeof(float));
  cudaMalloc(&d_output, (n - f + 1) * (n - f + 1) * sizeof(float));

  // Copy data to the device
  cudaMemcpy(d_input, input, n * n * sizeof(float), cudaMemcpyHostToDevice);

  // Dim - Adjust according to need
  dim3 block_dim(16, 16);

  // Call the kernel
  avg_pool_kernel<<<1, block_dim>>>(d_input, d_output, n, f);

  // Copy the result back to the CPU
  cudaMemcpy(output, d_output, (n - f + 1) * (n - f + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  // Synchronize here
  cudaDeviceSynchronize();

  // Free Device Memory
  cudaFree(d_input);
  cudaFree(d_output);

}

template <typename T = float>
inline T relu(T input) {
  return (input > 0 ? input : 0);
}

template <typename T = float>
inline T tanh(T input) {
  T z = std::exp(2 * input);
  return (z - 1.0) / (z + 1.0);
}


template <typename T = float>
void sigmoid(T *input, int n) {
  for (int i = 0; i < n; i += 1) {
    input[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }
}

template <typename T = float>
void softmax(T* input, int n) {
  T den = 0;
  for (int i = 0; i < n; i += 1) {
    input[i] = std::exp(input[i]);
    den += input[i];
  }

  for (int i = 0; i < n; i += 1) {
    input[i] /= den;
  }
}

template <typename T = float>
void read_matrix(T *matrix, int N) {
  for (int i = 0; i < N * N; i += 1) {
    std::cin >> matrix[i];
  }
}


int main(int argc, char** argv) {

  int idx = 1;
  int choice = std::stoi(argv[idx++]);
  
  // Convolution
  if (choice == 1) {
    int N, M, P;
    N = std::stoi(argv[idx++]);
    M = std::stoi(argv[idx++]);
    P = std::stoi(argv[idx++]);

    // std::vector<std::vector<float>> input_matrix(N, std::vector<float>(N));
    float *input_matrix = new float[N * N];
    for (int i = 0; i < N * N; i += 1) {
      input_matrix[i] = std::stof(argv[idx++]);
    }

    // std::vector<std::vector<float>> kernel_matrix(M, std::vector<float>(M));
    float *kernel_matrix = new float[M * M];
    for (int i = 0; i < M * M; i += 1) {
      kernel_matrix[i] = std::stof(argv[idx++]);
    }

    float *conv = new float[(N + 2 * P - M + 1) * (N + 2 * P - M + 1)];
    conv2D(input_matrix, kernel_matrix, conv, N, M, P);

    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < (N + 2 * P - M + 1); i += 1) {
      for (int j = 0; j < (N + 2 * P - M + 1); j += 1) {
        std::cout << conv[i * (N + 2 * P - M + 1) + j] << " ";
      }
      std::cout << std::endl;
    }

    delete[] input_matrix;
    delete[] kernel_matrix;
    delete[] conv;
  }
  // Relu or Tanh
  else if  (choice == 2) {
    int N, M, A;

    A = std::stoi(argv[idx++]);
    N = std::stoi(argv[idx++]);
    M = std::stoi(argv[idx++]);


    float *input_matrix = new float[N * M];
    for (int i = 0; i < N * M; i += 1) {
      input_matrix[i] = std::stof(argv[idx++]);
      input_matrix[i] = (A ? tanh(input_matrix[i]) : relu(input_matrix[i]));
    }


    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i += 1) {
      for (int j = 0; j < M; j += 1) {
        std::cout << input_matrix[i * M + j] << " ";
      }
      std::cout << std::endl;
    }

    delete[] input_matrix;
  }
  // Max pool or Avg Pool
  else if (choice == 3) {
    int N, M, A;

    A = std::stoi(argv[idx++]);
    M = std::stoi(argv[idx++]);
    N = std::stoi(argv[idx++]);

    float *input_matrix = new float[N * N];
    for (int i = 0; i < N * N; i += 1) {
      input_matrix[i] = std::stof(argv[idx++]);
    }

    float *output = new float[(N - M + 1) * (N - M + 1)];

    (A ? avg_pool(input_matrix, output, N, M) : max_pool(input_matrix, output, N, M)); 

    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N - M + 1; i += 1) {
      for (int j = 0; j < N - M + 1; j += 1) {
        std::cout << output[i * (N - M + 1) + j] << " ";
      }
      std::cout << std::endl;
    }

    delete[] input_matrix;
    delete[] output;
  }
  // Sigmoid or Softmax
  else if (choice == 4) {
    int A;
    A = std::stoi(argv[idx++]);
    float* input = new float[argc - 3];

    int N = argc - 3;

    for (int i = 0; i < N; i += 1){
      input[i] = std::stof(argv[idx++]);
    }

    (A ? softmax(input, N) : sigmoid(input, N));
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i += 1){
      std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    delete[] input;
  } 

  // input matrix
  // int N;
  // std::cin >> N;
  // float *input_matrix = new float[N * N];
  // read_matrix(input_matrix, N);

  // // kernel matrix
  // int f;
  // std::cin >> f;
  // float *kernel_matrix = new float[f * f];
  // read_matrix(kernel_matrix, f);

  // // padding
  // int pad;
  // std::cin >> pad;

  // // Get convolution.
  // float *conv = new float[(N + 2 * pad - f + 1) * (N + 2 * pad - f + 1)];
  // conv2D(input_matrix, kernel_matrix, conv, N, f, pad);

  // // for (int i = 0; i < (N + 2 * pad - f + 1) * (N + 2 * pad - f + 1); i += 1) {
  // //   std::cout << conv[i] << " ";
  // // }
  // // std::cout << std::endl;


  // // Activation Function - relu or tanh
  // std::string activation;
  // std::cin >> activation;

  // if (activation == "relu") {
  //   for (int i = 0; i < (N + 2 * pad - f + 1) * (N + 2 * pad - f + 1); i += 1) {
  //     conv[i] = relu(conv[i]);
  //   }
  // }
  // else if (activation == "tanh") {
  //   for (int i = 0; i < (N + 2 * pad - f + 1) * (N + 2 * pad - f + 1); i += 1) {
  //     conv[i] = tanh(conv[i]);
  //   }
  // }
  // else {
  //   std::cerr << "Specify correct activation function. Allowed are: relu or tanh." << std::endl;
  //   exit(0);
  // }

  // // pooling and its filter size
  // std::string pooling;
  // std::cin >> pooling;
  // int filter_size;
  // std::cin >> filter_size;


  // N = (N + 2 * pad - f + 1);
  // float* output = new float[(N / filter_size) * (N / filter_size)];

  // if (pooling == "max") {
  //   max_pool(conv, output, N, filter_size);
  // }
  // else if (pooling == "avg") {
  //   avg_pool(conv, output, N, filter_size);
  // }
  // else {
  //   std::cerr << "Specify correct pooling function. Allowed are: max or avg." << std::endl;
  //   exit(0);
  // }

  // std::string softorsig;
  // std::cin >> softorsig;

  // if (softorsig == "softmax") {
  //   softmax(output, (N / filter_size) * (N / filter_size));
  // }
  // else if (softorsig == "sigmoid") {
  //   sigmoid(output, (N / filter_size) * (N / filter_size));
  // }
  // else {
  //   std::cerr << "Specify either softmax or sigmoid." << std::endl;
  //   exit(0);
  // }

  // for (int i = 0; i < (N / filter_size) * (N / filter_size); i += 1) {
  //   std::cout << output[i] << " ";
  // }
  // std::cout << std::endl;

}