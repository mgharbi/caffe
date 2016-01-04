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
__global__ void RandomizeHSVForward(const int n, const Dtype* in, Dtype* out,
    Dtype rand_h, Dtype rand_s, Dtype rand_v,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width);
        int px = index % (height*width);
        int src_idx = blob_idx * height*width*chans // blob n index
            + px; // pixel index
        int chan_stride = height*width;

        Dtype h = in[src_idx];
        Dtype s = in[src_idx + chan_stride];
        Dtype v = in[src_idx + 2*chan_stride];

        out[src_idx+0*chan_stride] = fmod(Dtype(h+rand_h+1.0),Dtype(1.0));
        out[src_idx+1*chan_stride] = fmin(fmax(Dtype(s+rand_s),Dtype(0.0)), Dtype(1.0));
        out[src_idx+2*chan_stride] = fmin(fmax(Dtype(v+rand_v),Dtype(0.0)), Dtype(1.0));
    }
}

template <typename Dtype>
void RandomizeHSVLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int count = top[0]->count();

    // Process only during training
    float randomize;
    caffe_rng_uniform<float>(1, 0.0, 1.0, &randomize);
    LOG(INFO) << "GPU random " << randomized_ratio_;
    if (randomize < randomized_ratio_ || this->phase_ != TRAIN) {
        caffe_copy(count, bottom_data, top_data);
        return;
    }

    vector<int> shape = bottom[0]->shape();

    Dtype rand_h;
    Dtype rand_s;
    Dtype rand_v;
    caffe_rng_uniform<Dtype>(1, 0.0, 1.0, &rand_h);
    caffe_rng_uniform<Dtype>(1, -0.3, 0.3, &rand_s);
    caffe_rng_uniform<Dtype>(1, -0.3, 0.3, &rand_v);
    int npix = shape[0]*shape[2]*shape[3];
    RandomizeHSVForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            rand_h, rand_s, rand_v,
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomizeHSVLayer);

}  // namespace caffe
