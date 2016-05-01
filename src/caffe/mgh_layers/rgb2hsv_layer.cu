#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/rgb2hsv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RGB2HSVForward(const int n, const Dtype* in, Dtype* out,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width);
        int px = index % (height*width);
        int src_idx = blob_idx * height*width*chans // blob n index
            + px; // pixel index
        int chan_stride = height*width;

        Dtype r = in[src_idx + 0*chan_stride];
        Dtype g = in[src_idx + 1*chan_stride];
        Dtype b = in[src_idx + 2*chan_stride];

        Dtype maxi = fmax(r,fmax(g,b));
        Dtype mini = fmin(r,fmin(g,b));
        Dtype delta = maxi-mini;

        Dtype h = 0;
        Dtype v = 0; 
        Dtype s = 0; 

        // Value
        v = maxi;

        // Saturation
        if(v == 0 || delta == 0) {
            s = 0;
        }else{
            s = delta/v;
        }

        // Hue
        if(delta == 0){
            h = 0;
        }else {
            if(maxi == r) {
                h = (g-b)/delta;
            }else if(maxi == g) {
                h = 2.0 + (b-r)/delta;
            }else if(maxi == b) {
                h = 4.0 + (r-g)/delta;
            }
            h = fmod((h+6)/6.0,1.0);
        }

        out[src_idx+0*chan_stride] = h;
        out[src_idx+1*chan_stride] = s;
        out[src_idx+2*chan_stride] = v;
    }
}

template <typename Dtype>
void RGB2HSVLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[2]*shape[3];
    RGB2HSVForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void RGB2HSVLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(RGB2HSVLayer);

}  // namespace caffe
