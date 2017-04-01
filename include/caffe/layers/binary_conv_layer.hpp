#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
class BinaryConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit BinaryConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {
      }
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
   virtual vector<Dtype> compute_alfa_kernel(const Dtype* weight);
  virtual void compute_binary_weight(const Dtype* weight, Dtype* binary_weight, 
                                     vector<Dtype> kernel_alfa);
  virtual void update_weight_diff(const Dtype* weight, Dtype* weight_diff, 
                                  Dtype* binary_weight_diff, 
                                  vector<Dtype> kernel_alfa);
  virtual void scale_weight_diff(Dtype* weight_diff);
  shared_ptr<Blob<Dtype> > binary_weight_;
  int kernel_size_;
  int weight_size_;
  bool update_weight_diff_;
  bool scale_weight_diff_;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_CONV_LAYER_HPP_
