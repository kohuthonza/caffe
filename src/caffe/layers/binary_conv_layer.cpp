#include "caffe/layers/binary_conv_layer.hpp"

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
void BinaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  const vector<int> weight_shape = this->blobs_[0]->shape();
  binary_weight_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(weight_shape));
  weight_size_ = this->blobs_[0]->count();
  kernel_weight_size_ = weight_shape[1] * weight_shape[2] * weight_shape[3];
  update_weight_diff_ = this->layer_param_.binary_convolution_param().update_weight_diff();
  scale_weight_diff_ = this->layer_param_.binary_convolution_param().scale_weight_diff();
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* binary_weight = binary_weight_->mutable_cpu_data();
  compute_binary_weight(binary_weight, weight, compute_alfa_kernel(weight));
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
  const Dtype* weight = NULL;
  Dtype* binary_weight = NULL;
  Dtype* weight_diff = NULL;
  Dtype* binary_weight_diff = NULL;
  vector<Dtype> alfa_kernel;
  for (int i = 0; i < top.size(); ++i) {
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      weight = this->blobs_[0]->cpu_data();
      alfa_kernel = compute_alfa_kernel(weight);
      break;
    }
  }
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      binary_weight = binary_weight_->mutable_cpu_data();
      compute_binary_weight(binary_weight, weight, alfa_kernel);
      break;
    }
  }
  if (this->param_propagate_down_[0]) {
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    if (update_weight_diff_) {
      binary_weight_diff = binary_weight_->mutable_cpu_diff();
      std::fill(binary_weight_diff, binary_weight_diff + binary_weight_->count(), 0.);
    }
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
          if (update_weight_diff_) {
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
      if (this->param_propagate_down_[0]) {
        if (update_weight_diff_) {
          update_binary_weight_diff(binary_weight_diff, weight, alfa_kernel);
          if (scale_weight_diff_) {
            scale_binary_weight_diff(binary_weight_diff);
          }
          update_weight_diff(weight_diff, binary_weight_diff);
        }
      }
    }
  }
}

template <typename Dtype>
vector<Dtype> BinaryConvolutionLayer<Dtype>::compute_alfa_kernel(
      const Dtype* weight) {
  vector<Dtype> alfa_kernel;
  for (int i = 0; i < this->num_output_; ++i) {
    alfa_kernel.push_back(caffe_cpu_asum(kernel_weight_size_, weight +
                                         i * kernel_weight_size_) /
                                         kernel_weight_size_);
  }
  return alfa_kernel;
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_binary_weight(Dtype* binary_weight,
      const Dtype* weight, vector<Dtype> alfa_kernel) {
  for (int i = 0; i < weight_size_; ++i) {
    binary_weight[i] = copysign(1.0, weight[i]) * alfa_kernel[i / kernel_weight_size_];
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::update_binary_weight_diff(
      Dtype* binary_weight_diff, const Dtype* weight, vector<Dtype> alfa_kernel) {
  for (int i = 0; i < weight_size_; ++i) {
    if (weight[i] > -1. && weight[i] < 1.) {
      binary_weight_diff[i] *=  alfa_kernel[i / kernel_weight_size_] +
                                1. / kernel_weight_size_;
    }
    else {
      binary_weight_diff[i] *= 1. / kernel_weight_size_;
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::scale_binary_weight_diff(Dtype* binary_weight_diff) {
  for (int i = 0; i < weight_size_; ++i) {
    binary_weight_diff[i] *= kernel_weight_size_;
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::update_weight_diff(Dtype* weight_diff,
      Dtype* binary_weight_diff) {
    for (int i = 0; i < weight_size_; ++i) {
      weight_diff[i] += binary_weight_diff[i];
    }
}

#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);

}  // namespace caffe
