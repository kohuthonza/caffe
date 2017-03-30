#ifdef USE_CUDNN
#include <cmath>
#include <vector>

#include "caffe/layers/cudnn_binary_conv_layer.hpp"

namespace caffe {

__global__ void sync_binary_conv_groups() { }


template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* mutable_binary_weight = this->binary_weight_->mutable_cpu_data();
  this->compute_binary_weight(weight, mutable_binary_weight, this->compute_kernel_alfa(weight));
  const Dtype* binary_weight = this->binary_weight_->gpu_data();
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
  Dtype* weight_diff = NULL;
  Dtype* mutable_binary_weight = NULL;
  const Dtype* binary_weight = NULL;
  Dtype* binary_weight_diff = NULL;
  vector<Dtype> kernel_alfa;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    if (this->gradient_update_) {
      weight_diff = this->blobs_[0]->mutable_cpu_diff();
    }
    else {
      weight_diff = this->blobs_[0]->mutable_gpu_diff();
    }
    mutable_binary_weight = this->binary_weight_->mutable_cpu_data();
    kernel_alfa = this->compute_kernel_alfa(weight);
    this->compute_binary_weight(weight, mutable_binary_weight, kernel_alfa);
    binary_weight = this->binary_weight_->gpu_data();
    if (this->gradient_update_) {
      binary_weight_diff = this->binary_weight_->mutable_cpu_diff();
      for (int i = 0; i < this->binary_weight_->count(); ++i) {
        binary_weight_diff[i] = 0.;
      }
      binary_weight_diff = this->binary_weight_->mutable_gpu_diff();
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
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.

      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        if (this->gradient_update_) {
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle_[1*this->group_ + g],
                cudnn::dataType<Dtype>::one,
                bottom_descs_[i], bottom_data + bottom_offset_ * g,
                top_descs_[i],    top_diff + top_offset_ * g,
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
                top_descs_[i],    top_diff + top_offset_ * g,
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
          weight = this->blobs_[0]->cpu_data();
          mutable_binary_weight = this->binary_weight_->mutable_cpu_data();
          kernel_alfa = this->compute_kernel_alfa(weight);
          this->compute_binary_weight(weight, mutable_binary_weight, kernel_alfa);
          binary_weight = this->binary_weight_->gpu_data();
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
  if (this->gradient_update_) {
    binary_weight_diff = this->binary_weight_->mutable_cpu_diff();
    this->compute_binary_weight_diff(weight, weight_diff, binary_weight_diff, kernel_alfa);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBinaryConvolutionLayer);

}  // namespace caffe
#endif
