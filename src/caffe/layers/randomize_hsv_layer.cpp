#include <cfloat>
#include <cmath>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void RandomizeHSVLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to randomize_hsv layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int RandomizeHSVLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void RandomizeHSVLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void RandomizeHSVLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = top[0]->count();


    // Process only during training
    // if (this->phase_ != TRAIN) {
    //     caffe_copy(count, bottom_data, top_data);
    //     return;
    // }
    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) {
        Dtype rand_h;
        Dtype rand_s;
        Dtype rand_v;
        caffe_rng_uniform<Dtype>(1, 0.0, 1.0, &rand_h);
        caffe_rng_uniform<Dtype>(1, -0.3, 0.3, &rand_s);
        caffe_rng_uniform<Dtype>(1, -0.3, 0.3, &rand_v);

        for (int y = 0; y < shape[2]; ++y) 
        for (int x = 0; x < shape[3]; ++x) 
        {
            Dtype h = bottom_data[bottom[0]->offset(n,0,y,x)];
            Dtype s = bottom_data[bottom[0]->offset(n,1,y,x)];
            Dtype v = bottom_data[bottom[0]->offset(n,2,y,x)];

            top_data[top[0]->offset(n,0,y,x)] = std::fmod(h+rand_h+1.0,Dtype(1.0));
            top_data[top[0]->offset(n,1,y,x)] = std::min(std::max(s+rand_s,Dtype(0.0)), Dtype(1.0));
            top_data[top[0]->offset(n,2,y,x)] = std::min(std::max(v+rand_v,Dtype(0.0)), Dtype(1.0));
            // top_data[top[0]->offset(n,1,y,x)] = s;
            // top_data[top[0]->offset(n,2,y,x)] = v;
        }
    }

}

#ifdef CPU_ONLY
STUB_GPU(RandomizeHSVLayer);
#endif

INSTANTIATE_CLASS(RandomizeHSVLayer);
REGISTER_LAYER_CLASS(RandomizeHSV);

}  // namespace caffe
