#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/rot90_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void Rot90Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  CHECK_EQ(bottom[0]->shape()[2], bottom[0]->shape()[3]) <<  " Rot90 layer requires square patches";
}

template <typename Dtype>
void Rot90Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void Rot90Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

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
                top_data[top[0]->offset(n,z,y,x)] = bottom_data[bottom[0]->offset(n,z,x,y)];
            }
        }
    }

    delete[] randomize;
}

#ifdef CPU_ONLY
STUB_GPU(Rot90Layer);
#endif

INSTANTIATE_CLASS(Rot90Layer);
REGISTER_LAYER_CLASS(Rot90);

}  // namespace caffe
