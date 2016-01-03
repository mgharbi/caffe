#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void CenterImageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void CenterImageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CenterImageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int z = 0; z < shape[1]; ++z) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        Dtype val = bottom_data[bottom[0]->offset(n,z,y,x)];
        top_data[top[0]->offset(n,z,y,x)] = val - 0.5;
    }
}

template <typename Dtype>
void CenterImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int z = 0; z < shape[1]; ++z) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        Dtype val = top_diff[top[0]->offset(n,z,y,x)];
        bottom_diff[top[0]->offset(n,z,y,x)] = val;
    }
}

#ifdef CPU_ONLY
STUB_GPU(CenterImageLayer);
#endif

INSTANTIATE_CLASS(CenterImageLayer);
REGISTER_LAYER_CLASS(CenterImage);

}  // namespace caffe
