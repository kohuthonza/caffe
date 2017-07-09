#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>
#include <cmath>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class BinaryConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit BinaryConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BinaryConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  virtual void compute_binary_weight(Dtype* binary_weight, const Dtype* weight);
  virtual vector<Dtype> compute_alfa_kernel(const Dtype* weight);
  virtual void apply_alfa_kernel(Dtype *top_data, vector<Dtype> alfa_kernel,
                                 vector<int> top_shape);
  virtual void update_binary_weight_diff(Dtype* binary_weight_diff,
                                         const Dtype* weight,
                                         vector<Dtype> alfa_kernel);
  virtual void scale_binary_weight_diff(Dtype* binary_weight_diff);
  virtual void update_weight_diff(Dtype* weight_diff, Dtype* binary_weight_diff);
  shared_ptr<Blob<Dtype> > binary_weight_;
  int weight_size_;
  int kernel_weight_size_;
  bool update_weight_diff_;
  bool scale_weight_diff_;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_CONV_LAYER_HPP_
