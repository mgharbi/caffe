#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/permute_channels_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {


template <typename Dtype>
void PermuteChannelsLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
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

INSTANTIATE_LAYER_GPU_FUNCS(PermuteChannelsLayer);

}  // namespace caffe
