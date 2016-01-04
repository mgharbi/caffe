#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HSV2RGBForward(const int n, const Dtype* in, Dtype* out,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width);
        int px = index % (height*width);
        int src_idx = blob_idx * height*width*chans // blob n index
            + px; // pixel index
        int chan_stride = height*width;

        Dtype h = in[src_idx + 0*chan_stride];
        Dtype s = in[src_idx + 1*chan_stride];
        Dtype v = in[src_idx + 2*chan_stride];

        int hi = floor(h*6);
        Dtype f = h*6-hi;
        Dtype p = v*(1-s);
        Dtype q = v*(1-f*s);
        Dtype t = v*(1-(1-f)*s);

        Dtype r = 0;
        Dtype g = 0;
        Dtype b = 0;
        switch(hi%6) {
            case 0:
                r = v;
                g = t;
                b = p;
                break;
            case 1:
                r = q;
                g = v;
                b = p;
                break;
            case 2:
                r = p;
                g = v;
                b = t;
                break;
            case 3:
                r = p;
                g = q;
                b = v;
                break;
            case 4:
                r = t;
                g = p;
                b = v;
                break;
            case 5:
                r = v;
                g = p;
                b = q;
                break;
        }

        out[src_idx+0*chan_stride] = r;
        out[src_idx+1*chan_stride] = g;
        out[src_idx+2*chan_stride] = b;
    }
}

template <typename Dtype>
void HSV2RGBLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[2]*shape[3];
    HSV2RGBForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(HSV2RGBLayer);

}  // namespace caffe
