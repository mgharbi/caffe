#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/permute_channels_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void PermuteChannelsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to permute_channels layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));

    randomized_ratio_ = this->layer_param_.mgh_preprocessor_param().randomized_ratio();
}

template <typename Dtype>
int PermuteChannelsLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void PermuteChannelsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void PermuteChannelsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = top[0]->count();


    float randomize;
    caffe_rng_uniform<float>(1, 0.0, 1.0, &randomize);
    // Process only during training
    if (this->phase_ != TRAIN) {
        randomize = 1;
    }
    if(randomize < randomized_ratio_) { // flip a coin and permute channel
        vector<int> shape = bottom[0]->shape();
        int npix = shape[3]*shape[2];
        vector<int> chans(3);
        chans[0] = 0;
        chans[1] = 1;
        chans[2] = 2;
        shuffle(chans.begin(), chans.end());
        for (int n = 0; n < shape[0]; ++n) {
            caffe_copy(npix, bottom_data+bottom[0]->offset(n,0,0,0), top_data+top[0]->offset(n,chans[0],0,0));
            caffe_copy(npix, bottom_data+bottom[0]->offset(n,1,0,0), top_data+top[0]->offset(n,chans[1],0,0));
            caffe_copy(npix, bottom_data+bottom[0]->offset(n,2,0,0), top_data+top[0]->offset(n,chans[2],0,0));
        }
    }else {
        // Copy data over
        caffe_copy(count, bottom_data, top_data);
    }

}

#ifdef CPU_ONLY
STUB_GPU(PermuteChannelsLayer);
#endif

INSTANTIATE_CLASS(PermuteChannelsLayer);
REGISTER_LAYER_CLASS(PermuteChannels);

}  // namespace caffe
