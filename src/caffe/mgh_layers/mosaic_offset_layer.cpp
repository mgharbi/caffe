#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/mosaic_offset_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MosaicOffsetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to mosaic offset layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));

}

template <typename Dtype>
int MosaicOffsetLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void MosaicOffsetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MosaicOffsetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = top[0]->count();

    // Process only during training
    if (this->phase_ != TRAIN) {
        caffe_copy(count, bottom_data, top_data);
        return;
    }

    int offset_x = Rand(2);
    int offset_y = Rand(2);

    // No offset, copy data over
    if(offset_x == 0 && offset_y == 0) {
        caffe_copy(count, bottom_data, top_data);
        return;
    }

    vector<int> shape = bottom[0]->shape();

    for (int n = 0; n < shape[0]; ++n) 
    for (int c = 0; c < shape[1]; ++c) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        top_data[top[0]->offset(n,c,y,x)] = bottom_data[bottom[0]->offset(n,c,std::max(y-offset_y,0),std::max(x-offset_x,0))];
    }
}

#ifdef CPU_ONLY
STUB_GPU(MosaicOffsetLayer);
#endif

INSTANTIATE_CLASS(MosaicOffsetLayer);
REGISTER_LAYER_CLASS(MosaicOffset);

}  // namespace caffe
