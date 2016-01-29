#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/srgb2linear_rgb_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SRGB2LinearRGBForward(const int n, const Dtype* in, Dtype* out,
    int height, int width, int chans)
{
    float t_ = 0.04045 ;
    float a_ = 0.055;
    float b_ = 1+a_;
    float c_ = 2.4;
    float d_ = 12.92;
    CUDA_KERNEL_LOOP(index, n) {
        Dtype px = in[index];
        if ( px > t_ ) {
            px = pow(( px + a_ ) / b_ , c_);
        } else {
            px = px / d_;
        }  
        out[index] = px;
    }
}

template <typename Dtype>
__global__ void SRGB2LinearRGBBackward(const int n, const Dtype* top_diff, Dtype* bottom_diff, const Dtype* bottom_data,
    int height, int width, int chans)
{
    float t_ = 0.04045 ;
    float a_ = 0.055;
    float b_ = 1+a_;
    float c_ = 2.4;
    float d_ = 12.92;
    CUDA_KERNEL_LOOP(index, n) {
        Dtype dx = top_diff[index];

        Dtype px = bottom_data[index];

        Dtype dr = Dtype(0);

        if ( px > t_) {
            dr = c_/b_*pow(( px + a_ ) / b_  , c_-1);
        } else {
            dr = 1.0/d_;
        }  

        bottom_diff[index] = dr*dx;
    }
}

template <typename Dtype>
void SRGB2LinearRGBLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    SRGB2LinearRGBForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void SRGB2LinearRGBLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff    = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff       = bottom[0]->mutable_cpu_diff();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    SRGB2LinearRGBBackward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, top_diff, bottom_diff, bottom_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SRGB2LinearRGBLayer);

}  // namespace caffe
