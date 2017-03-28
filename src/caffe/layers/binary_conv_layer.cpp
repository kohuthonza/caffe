#include <cmath>
#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

#include <iostream>
using namespace std;

namespace caffe {
      

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_binary_weight(const Dtype* weight, Dtype* binary_weight, vector<Dtype> kernel_alfa) {
  for (int i = 0; i < this->num_output_; ++i){
    for (int j = 0; j < this->kernel_size_; ++j){
      binary_weight[i * this->kernel_size_ + j] = copysign(1.0, weight[i * this->kernel_size_ + j]) * kernel_alfa[i];
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_binary_weight_diff(const Dtype* weight, Dtype* weight_diff, Dtype* binary_weight_diff, vector<Dtype> kernel_alfa) {
  for (int i = 0; i < this->num_output_; ++i) {
    for (int j = 0; j < this->kernel_size_; ++j) {
      if (weight[i * this->kernel_size_ + j] < 1. && weight[i * this->kernel_size_ + j] > -1.) {
        binary_weight_diff[i * this->kernel_size_ + j] = binary_weight_diff[i * this->kernel_size_ + j] * (kernel_alfa[i] + 1. / this->kernel_size_); 
      }
      else {
        binary_weight_diff[i * this->kernel_size_ + j] = binary_weight_diff[i * this->kernel_size_ + j] * (1. / this->kernel_size_);
      }
    } 
  }
  for (int i = 0; i < this->blobs_[0]->count(); ++i) {
    weight_diff[i] += binary_weight_diff[i];
  }
}

template <typename Dtype>
vector<Dtype> BinaryConvolutionLayer<Dtype>::compute_kernel_alfa(const Dtype* weight) {
  vector<Dtype> kernel_alfa;
  for (int i = 0; i < this->num_output_; ++i) {
    kernel_alfa.push_back(caffe_cpu_asum(this->kernel_size_, weight + i * this->kernel_size_) / this->kernel_size_);
  }
  return kernel_alfa;
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top){
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  vector<int> kernel_blob_shape;
  kernel_blob_shape.push_back(this->blobs_[0]->shape(0));
  kernel_blob_shape.push_back(this->blobs_[0]->shape(1));
  kernel_blob_shape.push_back(this->blobs_[0]->shape(2));
  kernel_blob_shape.push_back(this->blobs_[0]->shape(3));
  this->binary_weight_ = new Blob<Dtype>(kernel_blob_shape);
  this->kernel_size_ = this->blobs_[0]->shape(1) * this->blobs_[0]->shape(2) * this->blobs_[0]->shape(3);
  this->gradient_update_ = this->layer_param_.binary_convolution_param().gradient_update();
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* binary_weight = this->binary_weight_->mutable_cpu_data();
  compute_binary_weight(weight, binary_weight, compute_kernel_alfa(weight));
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, binary_weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* binary_weight = this->binary_weight_->mutable_cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* binary_weight_diff = this->binary_weight_->mutable_cpu_diff();
  vector<Dtype> kernel_alfa = compute_kernel_alfa(weight);  
  compute_binary_weight(weight, binary_weight, kernel_alfa);
  for (int i = 0; i < this->binary_weight_->count(); ++i) {
      binary_weight_diff[i] = 0.;
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          if (this->gradient_update_) {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, binary_weight_diff); 
          }
          else {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);    
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, binary_weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
      if (this->gradient_update_) {
        compute_binary_weight_diff(weight, weight_diff, binary_weight_diff, kernel_alfa);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);

}  // namespace caffe
