#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/linear_rgb2srgb_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LinearRGB2SRGBForward(const int n, const Dtype* in, Dtype* out,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        Dtype px = in[index];

        if ( px > 0.0031308 ) {
            px = 1.055*pow(px,Dtype(1.0/2.4)) - 0.055;
        } else {
            px = px * 12.92;
        }  

        out[index] = px;
    }
}

template <typename Dtype>
__global__ void LinearRGB2SRGBBackward(const int n, const Dtype* top_diff, Dtype* bottom_diff, const Dtype* bottom_data,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        Dtype dx = top_diff[index];

        Dtype px = bottom_data[index];

        Dtype dr = Dtype(0);

        if ( px > 0.0031308 ) {
            dr = 1.055*(1.0/2.4)*pow(px,Dtype(1.0/2.4-1)) ;
        } else {
            dr = 12.92;
        }  

        bottom_diff[index] = dr*dx;
    }
}

template <typename Dtype>
void LinearRGB2SRGBLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    LinearRGB2SRGBForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void LinearRGB2SRGBLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff    = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff       = bottom[0]->mutable_cpu_diff();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    LinearRGB2SRGBBackward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, top_diff, bottom_diff, bottom_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(LinearRGB2SRGBLayer);

}  // namespace caffe
