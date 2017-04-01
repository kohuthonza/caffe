#ifdef USE_CUDNN
#include <cmath>
#include <vector>

#include "caffe/layers/cudnn_binary_conv_layer.hpp"

#include <iostream>
using namespace std;

namespace caffe {

__global__ void sync_binary_conv_groups() { }

template <typename Dtype>
__global__ void compute_binary_weight(const int n, const Dtype* weight,
               Dtype* binary_weight, const Dtype* alfa_kernel, const int kernel_size) {
  CUDA_KERNEL_LOOP(index, n) {
    binary_weight[index] = copysign(1.0, weight[index]) * alfa_kernel[index / kernel_size];
  }
}

template <typename Dtype>
__global__ void update_weight_diff(const int n, const Dtype* weight,
              Dtype* weight_diff, Dtype* binary_weight_diff,
              const Dtype* alfa_kernel, const int kernel_size) {
  CUDA_KERNEL_LOOP(index, n) {
    if (weight[index] < 1. && weight[index] > -1.) {
      weight_diff[index] += binary_weight_diff[index] *
                            (alfa_kernel[index / kernel_size] + 1./kernel_size);
    }
    else {
      weight_diff[index] += binary_weight_diff[index] * 1./kernel_size;
    }
  }
}

template <typename Dtype>
__global__ void scale_weight_diff(const int n, Dtype* weight_diff,
              const int kernel_size) {
  CUDA_KERNEL_LOOP(index, n) {
    weight_diff[index] *= kernel_size;
  }
}

template <typename Dtype>
__global__ void copy_abs_value(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = abs(in[index]);
  }
}

template <typename Dtype>
__global__ void set_values_to_zero(const int n, Dtype* in) {
  CUDA_KERNEL_LOOP(index, n) {
    in[index] = 0.;
  }
}

template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* abs_weight = abs_weight_->mutable_gpu_data();
  copy_abs_value<<<CAFFE_GET_BLOCKS(this->weight_size_),
                   CAFFE_CUDA_NUM_THREADS>>>
                   (this->weight_size_, weight, abs_weight);
  const Dtype* alfa_kernel_multiplier = alfa_kernel_multiplier_->gpu_data();
  Dtype* alfa_kernel = alfa_kernel_->mutable_gpu_data();
  CUDNN_CHECK(cudnnConvolutionForward(*alfa_handle_,
        cudnn::dataType<Dtype>::one,
        alfa_bottom_desc_, abs_weight,
        alfa_filter_desc_, alfa_kernel_multiplier,
        alfa_conv_desc_,
        *alfa_fwd_algo_, NULL, 0,
        cudnn::dataType<Dtype>::zero,
        alfa_top_desc_, alfa_kernel));
  Dtype* binary_weight = this->binary_weight_->mutable_gpu_data();
  compute_binary_weight<<<CAFFE_GET_BLOCKS(this->weight_size_),
                                     CAFFE_CUDA_NUM_THREADS>>>
                                     (this->weight_size_, weight, binary_weight,
                                     alfa_kernel, this->kernel_size_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, binary_weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_binary_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* abs_weight = NULL;
  Dtype* weight_diff = NULL;
  Dtype* binary_weight = NULL;
  Dtype* binary_weight_diff = NULL;
  Dtype* alfa_kernel = NULL;
  const Dtype* alfa_kernel_multiplier = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    abs_weight = abs_weight_->mutable_gpu_data();
    copy_abs_value<<<CAFFE_GET_BLOCKS(this->weight_size_),
                     CAFFE_CUDA_NUM_THREADS>>>
                     (this->weight_size_, weight, abs_weight);
    alfa_kernel_multiplier = alfa_kernel_multiplier_->gpu_data();
    alfa_kernel = alfa_kernel_->mutable_gpu_data();
    CUDNN_CHECK(cudnnConvolutionForward(*alfa_handle_,
          cudnn::dataType<Dtype>::one,
          alfa_bottom_desc_, abs_weight,
          alfa_filter_desc_, alfa_kernel_multiplier,
          alfa_conv_desc_,
          *alfa_fwd_algo_, NULL, 0,
          cudnn::dataType<Dtype>::zero,
          alfa_top_desc_, alfa_kernel));
    binary_weight = this->binary_weight_->mutable_gpu_data();
    compute_binary_weight<<<CAFFE_GET_BLOCKS(this->weight_size_),
                            CAFFE_CUDA_NUM_THREADS>>>
                            (this->weight_size_, weight, binary_weight,
                             alfa_kernel, this->kernel_size_);
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    if (this->update_weight_diff_) {
      binary_weight_diff = this->binary_weight_->mutable_gpu_diff();
      set_values_to_zero<<<CAFFE_GET_BLOCKS(this->weight_size_),
                           CAFFE_CUDA_NUM_THREADS>>>
                           (this->weight_size_, binary_weight_diff);
    }
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.

      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        if (this->update_weight_diff_) {
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle_[1*this->group_ + g],
                cudnn::dataType<Dtype>::one,
                bottom_descs_[i], bottom_data + bottom_offset_ * g,
                top_descs_[i], top_diff + top_offset_ * g,
                conv_descs_[i],
                bwd_filter_algo_[i], workspace[1*this->group_ + g],
                workspace_bwd_filter_sizes_[i],
                cudnn::dataType<Dtype>::one,
                filter_desc_, binary_weight_diff + this->weight_offset_ * g));
        }
        else {
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle_[1*this->group_ + g],
                cudnn::dataType<Dtype>::one,
                bottom_descs_[i], bottom_data + bottom_offset_ * g,
                top_descs_[i], top_diff + top_offset_ * g,
                conv_descs_[i],
                bwd_filter_algo_[i], workspace[1*this->group_ + g],
                workspace_bwd_filter_sizes_[i],
                cudnn::dataType<Dtype>::one,
                filter_desc_, weight_diff + this->weight_offset_ * g));
        }
      }


      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          const Dtype* weight = this->blobs_[0]->gpu_data();
          Dtype* abs_weight = abs_weight_->mutable_gpu_data();
          copy_abs_value<<<CAFFE_GET_BLOCKS(this->weight_size_),
                           CAFFE_CUDA_NUM_THREADS>>>
                           (this->weight_size_, weight, abs_weight);
          const Dtype* alfa_kernel_multiplier = alfa_kernel_multiplier_->gpu_data();
          Dtype* alfa_kernel = alfa_kernel_->mutable_gpu_data();
          CUDNN_CHECK(cudnnConvolutionForward(*alfa_handle_,
                cudnn::dataType<Dtype>::one,
                alfa_bottom_desc_, abs_weight,
                alfa_filter_desc_, alfa_kernel_multiplier,
                alfa_conv_desc_,
                *alfa_fwd_algo_, NULL, 0,
                cudnn::dataType<Dtype>::zero,
                alfa_top_desc_, alfa_kernel));
          Dtype* binary_weight = this->binary_weight_->mutable_gpu_data();
          compute_binary_weight<<<CAFFE_GET_BLOCKS(this->weight_size_),
                                  CAFFE_CUDA_NUM_THREADS>>>
                                  (this->weight_size_, weight, binary_weight,
                                   alfa_kernel, this->kernel_size_);
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, binary_weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_binary_conv_groups<<<1, 1>>>();
  }
  if (this->update_weight_diff_) {
    update_weight_diff<<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
                         CAFFE_CUDA_NUM_THREADS>>>
                         (this->weight_size_, weight, weight_diff,
                          binary_weight_diff, alfa_kernel, this->kernel_size_);
    if (this->scale_weight_diff_) {
      scale_weight_diff<<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
                          CAFFE_CUDA_NUM_THREADS>>>
                          (this->weight_size_, weight_diff, this->kernel_size_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBinaryConvolutionLayer);

}  // namespace caffe
#endif
