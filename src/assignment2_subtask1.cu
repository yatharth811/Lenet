#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>


template <typename T = float>
std::vector<std::vector<T>> conv2D(const std::vector<std::vector<T>> &input, const std::vector<std::vector<T>> &kernel, int pad)
{
  int f = kernel.size();
  int n = input.size();
  std::vector<std::vector<T>> ninput(n + 2 * pad, std::vector<T>(n + 2 * pad, 0));

  // Build padded matrix
  for (int i = 0; i < n; i += 1) {
    for (int j = 0; j < n; j += 1) {
      ninput[i + pad][j + pad] = input[i][j];
    }
  }

  // do convolution with new matrix here
  n += 2 * pad;
  std::vector<std::vector<T>> conv(n - f + 1, std::vector<T> (n - f + 1, 0));
  for (int i = 0; i < n - f + 1; i += 1) {
    for (int j = 0; j < n - f + 1; j += 1) {
      for (int k = 0; k < f; k += 1) {
        for (int l = 0; l < f; l += 1) {
          conv[i][j] += kernel[k][l] * ninput[k + i][l + j];
        }
      }
    }
  }

  return conv;
}

template <typename T = float>
inline T relu(T input) {
  return (input > 0 ? input : 0);
}

template <typename T = float>
inline void relu(std::vector<std::vector<T>> &input) {
  int n = input.size(),m=input[0].size();
  for (int i = 0; i < n; i += 1) {
    for (int j = 0; j < m; j += 1)
      input[i][j] = relu(input[i][j]);
  }
}

template <typename T = float>
inline T tanh(T input) {
  T z = std::exp(2 * input);
  return (z - 1.0) / (z + 1.0);
}

template <typename T = float>
inline void tanh(std::vector<std::vector<T>> &input) {
  int n = input.size(),m=input[0].size();
  for (int i = 0; i < n; i += 1) {
    for (int j = 0; j < m; j += 1)
      input[i][j] = tanh(input[i][j]);
  }
}

template <typename T = float>
std::vector<std::vector<T>> max_pool(const std::vector<std::vector<T>> &input, int filter_size) {
  int n = input.size();
  int f = filter_size;
  std::vector<std::vector<T>> output(n - f + 1, std::vector<T>(n - f + 1, std::numeric_limits<T>::lowest()));
  for (int i = 0; i < n - f + 1; i += 1) {
    for (int j = 0; j < n - f + 1; j += 1) {
      T mx = output[i][j];
      for (int k = 0; k < f; k += 1) {
        for (int l = 0; l < f; l += 1) {
          mx = std::max(mx, input[i + k][j + l]);
        }
      }
      output[i][j] = mx;
    }
  }

  return output;
}

template <typename T = float>
std::vector<std::vector<T>> avg_pool(const std::vector<std::vector<T>> &input, int filter_size) {
  int n = input.size();
  int f = filter_size;
  std::vector<std::vector<T>> output(n - f + 1, std::vector<T>(n - f + 1, 0));
  for (int i = 0; i < n - f + 1; i += 1) {
    for (int j = 0; j < n - f + 1; j += 1) {
      T sum{};
      for (int k = 0; k < f; k += 1) {
        for (int l = 0; l < f; l += 1) {
          sum += input[i + k][j + l];
        }
      }
      output[i][j] = 1.0 * sum / (f * f);
    }
  }

  return output;
}

template <typename T = float>
std::vector<T> sigmoid(const std::vector<T> &input) {
  int n = input.size();
  std::vector<T> output(n);
  for (int i = 0; i < n; i += 1) {
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }

  return output;
}

template <typename T = float>
std::vector<T> softmax(const std::vector<T> &input) {
  int n = input.size();
  std::vector<T> output(n);

  T den = 0;
  for (int i = 0; i < n; i += 1) {
    output[i] = std::exp(input[i]);
    den += output[i];
  }

  for (int i = 0; i < n; i += 1) {
    output[i] /= den;
  }

  return output;
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

    std::vector<std::vector<float>> input_matrix(N, std::vector<float>(N));
    for (int i = 0; i < N; i += 1) {
      for (int j = 0; j < N; j += 1) {
        input_matrix[i][j] = std::stof(argv[idx++]);
      }
    }

    std::vector<std::vector<float>> kernel_matrix(M, std::vector<float>(M));
    for (int i = 0; i < M; i += 1) {
      for (int j = 0; j < M; j += 1) {
        kernel_matrix[i][j] = std::stof(argv[idx++]);
      }
    }

    auto conv = conv2D(input_matrix, kernel_matrix, P);

    std::cout << std::fixed << std::setprecision(6);
    for (auto x: conv) {
      for (auto y: x) {
        std::cout << y << " ";
      }
      std::cout << std::endl;
    }

  }
  // Relu or Tanh
  else if  (choice == 2) {
    int N, M, A;

    A = std::stoi(argv[idx++]);
    N = std::stoi(argv[idx++]);
    M = std::stoi(argv[idx++]);
    
    std::vector<std::vector<float>> input_matrix(N, std::vector<float>(M));
    for (int i = 0; i < N; i += 1) {
      for (int j = 0; j < M; j += 1) {
        input_matrix[i][j] = std::stof(argv[idx++]);
      }
    }

    (A ? tanh(input_matrix) : relu(input_matrix)); 

    std::cout << std::fixed << std::setprecision(6);
    for (auto x: input_matrix) {
      for (auto y: x) {
        std::cout << y << " ";
      }
      std::cout << std::endl;
    }

  }
  // Max pool or Avg Pool
  else if (choice == 3) {
    int N, M, A;

    A = std::stoi(argv[idx++]);
    M = std::stoi(argv[idx++]);
    N = std::stoi(argv[idx++]);
    
    std::vector<std::vector<float>> input_matrix(N, std::vector<float>(N));
    for (int i = 0; i < N; i += 1) {
      for (int j = 0; j < N; j += 1) {
        input_matrix[i][j] = std::stof(argv[idx++]);
      }
    }

    auto output = (A ? avg_pool(input_matrix, M) : max_pool(input_matrix, M)); 

    std::cout << std::fixed << std::setprecision(6);
    for (auto x: output) {
      for (auto y: x) {
        std::cout << y << " ";
      }
      std::cout << std::endl;
    }

  }
  // Sigmoid or Softmax
  else if (choice == 4) {
    int A;
    A = std::stoi(argv[idx++]);
    std::vector<float> input;

    while (idx < argc) {
      input.emplace_back(std::stof(argv[idx]));
      idx += 1;
    }

    auto output = (A ? softmax(input) : sigmoid(input));
    std::cout << std::fixed << std::setprecision(6);
    for (auto x: output) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  } 

}