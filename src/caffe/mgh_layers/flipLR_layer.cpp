#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/flipLR_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void FlipLRLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void FlipLRLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FlipLRLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = top[0]->count();

    // Process only during training
    if (this->phase_ != TRAIN) {
        caffe_copy(count, bottom_data, top_data);
        return;
    }

    vector<int> shape = bottom[0]->shape();
    int w = shape[3];

    float* randomize = new float[shape[0]];
    caffe_rng_uniform<float>(shape[0], 0.0, 1.0, randomize);

    for (int n = 0; n < shape[0]; ++n) {
        if(randomize[n] > 0.5f) {
            for (int z = 0; z < shape[1]; ++z) 
            for (int y = 0; y < shape[2]; ++y) 
            for (int x = 0; x < shape[3]; ++x) 
            {
                top_data[top[0]->offset(n,z,y,x)] = bottom_data[bottom[0]->offset(n,z,y,x)];
            }
        }else {
            for (int z = 0; z < shape[1]; ++z) 
            for (int y = 0; y < shape[2]; ++y) 
            for (int x = 0; x < shape[3]; ++x) 
            {
                top_data[top[0]->offset(n,z,y,x)] = bottom_data[bottom[0]->offset(n,z,y,w-1-x)];
            }
        }
    }

    delete[] randomize;
}

#ifdef CPU_ONLY
STUB_GPU(FlipLRLayer);
#endif

INSTANTIATE_CLASS(FlipLRLayer);
REGISTER_LAYER_CLASS(FlipLR);

}  // namespace caffe
