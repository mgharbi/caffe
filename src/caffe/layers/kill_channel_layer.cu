#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void KillChannelLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
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
            caffe_gpu_memset(npix*sizeof(Dtype),Dtype(0),top_data+top[0]->offset(n,chanToKill,0,0));
        }
    }

}
INSTANTIATE_LAYER_GPU_FUNCS(KillChannelLayer);

}  // namespace caffe
