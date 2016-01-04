#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void KillChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to kill_channel layer should have 3 channels" ;

    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));

    randomized_ratio_ = this->layer_param_.mgh_preprocessor_param().randomized_ratio();
}

template <typename Dtype>
int KillChannelLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void KillChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void KillChannelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = top[0]->count();

    // Copy data over
    if(top[0] != bottom[0]) {
        caffe_copy(count, bottom_data, top_data);
    }

    // Process only during training
    if (this->phase_ != TRAIN) {
        return;
    }

    float randomize;
    caffe_rng_uniform<float>(1, 0.0, 1.0, &randomize);
    if(randomize < randomized_ratio_) { // flip a coin and permute channel
        int chanToKill = Rand(3);
        vector<int> shape = bottom[0]->shape();

        int npix = shape[3]*shape[2];

        for (int n = 0; n < shape[0]; ++n) 
        {
            caffe_set(npix,Dtype(0),top_data+top[0]->offset(n,chanToKill,0,0));
        }
    }

}

#ifdef CPU_ONLY
STUB_GPU(KillChannelLayer);
#endif

INSTANTIATE_CLASS(KillChannelLayer);
REGISTER_LAYER_CLASS(KillChannel);

}  // namespace caffe
