#pragma once
#include <iostream>
#include <limits>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

template<typename T = float>
int load_weights(T *weights, T *bias, int num_weights, int num_bias, std::string file_name) {

  std::fstream file(file_name);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_name << std::endl;
    return 0;
  }
	
  // Read the weights from the file
  for (int i = 0; i < num_weights; i += 1) {
    file >> weights[i];
  }
				
  for (int i = 0; i < num_bias; i += 1) {
    file >> bias[i];
  }
		

  // Close the file
  file.close();
  
  return 1;
}

std::vector<std::string> load_image_paths(std::string filename) {

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return {};
  }

  std::vector<std::string> img_paths;

  while (!file.eof()) {
    std::string s;
    file >> s;
    if(s.length()==0) continue;
    img_paths.push_back(s);
  }	
  
  return img_paths;
}

std::vector<int> load_labels(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    std::vector<int> labels;
    int label;
    while (file >> label) {
        labels.push_back(label);
    }

    file.close();

    return labels;
}


template<typename T = float>
int load_image(T *input, int n, std::string file_name) {

  std::fstream file(file_name);
  if (!file.is_open()) {
    std::cerr << "Failed to open image: " << file_name << std::endl;
    return 0;
  }
	
  // Read the weights from the file
  for (int i = 0; i < n; i += 1) {
    file >> input[i];
  }		

  // Close the file
  file.close();
  
  return 1;
}


template <typename T = float>
void max_pool(T* input, T* output, int n, int filter_size) {
  
  int f = filter_size;
  for (int i = 0, i_i = 0; i < n; i += f, i_i += 1) {
    for (int j = 0, j_j = 0; j < n; j += f, j_j += 1) {
      T mx = std::numeric_limits<T>::lowest();
      for (int k = 0; k < f; k += 1) {
        for (int l = 0; l < f; l += 1) {
          mx = std::max(mx, input[(i + k) * n + (j + l)]);
        }
	    }
	    output[i_i * (n / f) + j_j] = mx;
    }
  }

}

std::vector<float> top5_prob(float *a, int n){
    float prob[10];
    float sum = 0.0f;
    for(int i=0; i<10; i++) sum+=exp(a[i]);
    for(int i=0; i<10; i++) prob[i] = exp(a[i])/sum;
    std::sort(prob,prob+10);
    std::vector<float> result;
    for(int i=9; i>=5; i--) result.push_back(prob[i]);
    return result;
}

template<typename T>
bool writeToFile(const std::vector<T>& vec, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    for (const auto& elem : vec) {
        outFile << elem << std::endl;
    }

    if (!outFile.good()) {
        std::cerr << "Error writing to file: " << filename << std::endl;
        return false;
    }

    outFile.close();
    return true;
}

std::string get_img_name(std::string imgname){
  int ln = imgname.length();
  return imgname.substr(ln-11);
}
