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
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
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
    caffe_copy(count, bottom_data, top_data);

    // Process only during training
    if (this->phase_ != TRAIN) {
        return;
    }

    int killChan = Rand(2);
    if(killChan == 0) { // flip a coin and kill a channel
        int chanToKill = Rand(4);
        vector<int> shape = bottom[0]->shape();
        for (int n = 0; n < shape[0]; ++n) 
        for (int y = 0; y < shape[2]; ++y) 
        for (int x = 0; x < shape[3]; ++x) 
        {
            top_data[top[0]->offset(n,chanToKill,y,x)] = Dtype(0);
        }
    }

}

#ifdef CPU_ONLY
STUB_GPU(KillChannelLayer);
#endif

INSTANTIATE_CLASS(KillChannelLayer);
REGISTER_LAYER_CLASS(KillChannel);

}  // namespace caffe
