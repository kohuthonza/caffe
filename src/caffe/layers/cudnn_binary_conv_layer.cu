#ifdef USE_CUDNN
#include "caffe/layers/cudnn_binary_conv_layer.hpp"

namespace caffe {

__global__ void sync_binary_conv_groups() { }

template <typename Dtype>
__global__ void compute_binary_weight(Dtype* binary_weight, const Dtype* weight,
              const int weight_size) {
  CUDA_KERNEL_LOOP(index, weight_size) {
    if (weight[index] > 0.0)
    {
      binary_weight[index] = 1.0;
    }
    else
    {
      binary_weight[index] = -1.0;
    }
  }
}

template <typename Dtype>
__global__ void update_weight_diff(Dtype* weight_diff, Dtype* binary_weight_diff,
              const int weight_size) {
  CUDA_KERNEL_LOOP(index, weight_size) {
    weight_diff[index] += binary_weight_diff[index];
  }
}

template <typename Dtype>
__global__ void apply_alfa_kernel(Dtype* top_data, Dtype* alfa_kernel,
              const int batch_size, const int top_size,
              const int top_channel_size) {
  CUDA_KERNEL_LOOP(index, batch_size) {
    top_data[index] *= alfa_kernel[(index % top_size) / top_channel_size];
  }
}

template <typename Dtype>
__global__ void update_binary_weight_diff(Dtype* binary_weight_diff,
              const Dtype* weight, const Dtype* alfa_kernel,
              const int weight_size, const int kernel_weight_size) {
  CUDA_KERNEL_LOOP(index, weight_size) {
    if (weight[index] > -1. && weight[index] < 1.) {
      binary_weight_diff[index] *= alfa_kernel[index / kernel_weight_size]
                                   + 1./kernel_weight_size;
    }
    else {
      binary_weight_diff[index] *= 1./kernel_weight_size;
    }
  }
}

template <typename Dtype>
__global__ void scale_binary_weight_diff(Dtype* binary_weight_diff,
              const int weight_size, const int kernel_weight_size) {
  CUDA_KERNEL_LOOP(index, weight_size) {
    binary_weight_diff[index] *= kernel_weight_size;
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
  // Compute alfa with convolution
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
                                     (binary_weight, weight, this->weight_size_);
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
    apply_alfa_kernel<<<CAFFE_GET_BLOCKS(top[i]->count()),
                        CAFFE_CUDA_NUM_THREADS>>>
                        (top_data, alfa_kernel, top[i]->count(),
                         top[i]->shape(1) * top[i]->shape(2) * top[i]->shape(3),
                         top[i]->shape(2) * top[i]->shape(3));
  }
}

template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* abs_weight = NULL;
  Dtype* binary_weight = NULL;
  Dtype* weight_diff = NULL;
  Dtype* binary_weight_diff = NULL;
  Dtype* alfa_kernel = NULL;
  const Dtype* alfa_kernel_multiplier = NULL;
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i] || this->param_propagate_down_[0]) {
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
      break;
    }
  }
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      binary_weight = this->binary_weight_->mutable_gpu_data();
      compute_binary_weight<<<CAFFE_GET_BLOCKS(this->weight_size_),
                                         CAFFE_CUDA_NUM_THREADS>>>
                                         (binary_weight, weight,
                                          this->weight_size_);
      break;
    }
  }
  if (this->param_propagate_down_[0]) {
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
    shared_ptr<Blob<Dtype> > binary_top;
    Dtype *binary_top_diff = NULL;
    if (propagate_down[i]) {
      binary_top = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(top[i]->shape()));
      binary_top_diff = binary_top.get()->mutable_gpu_diff();
      caffe_gpu_memcpy(top[i]->count() * sizeof(Dtype), top_diff, binary_top_diff);
      apply_alfa_kernel<<<CAFFE_GET_BLOCKS(top[i]->count()),
                          CAFFE_CUDA_NUM_THREADS>>>
                          (binary_top_diff, alfa_kernel,
                           top[i]->count(), top[i]->shape(1) *
                           top[i]->shape(2) * top[i]->shape(3),
                           top[i]->shape(2) * top[i]->shape(3));
    }
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
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, binary_weight + this->weight_offset_ * g,
              top_descs_[i], binary_top_diff + top_offset_ * g,
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
  if (this->param_propagate_down_[0]) {
    if (this->update_weight_diff_) {
      update_binary_weight_diff<<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
                                  CAFFE_CUDA_NUM_THREADS>>>
                                  (binary_weight_diff, weight, alfa_kernel,
                                   this->weight_size_, this->kernel_weight_size_);
      if (this->scale_weight_diff_) {
        scale_binary_weight_diff<<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
                                   CAFFE_CUDA_NUM_THREADS>>>
                                   (binary_weight_diff, this->weight_size_,
                                    this->kernel_weight_size_);
      }
      update_weight_diff<<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
                           CAFFE_CUDA_NUM_THREADS>>>
                           (weight_diff, binary_weight_diff, this->weight_size_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBinaryConvolutionLayer);

}  // namespace caffe
#endif
