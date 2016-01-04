#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RGB2XYZForward(const int n, const Dtype* in, Dtype* out,
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

        if ( r > 0.04045 ) {
            r = pow(Dtype(( r + 0.055 ) / 1.055) , Dtype(2.4));
        } else {
            r = r / 12.92;
        }  
        if ( g > 0.04045 ) {
            g = pow(Dtype(( g + 0.055 ) / 1.055) , Dtype(2.4));
        } else {
            g = g / 12.92;
        }  
        if ( b > 0.04045 ) {
            b = pow(Dtype(( b + 0.055 ) / 1.055) , Dtype(2.4));
        } else {
            b = b / 12.92;
        }  

        //Observer. = 2°, Illuminant = D65
        Dtype X = r * 0.412453 + g * 0.357580 + b * 0.180423;
        Dtype Y = r * 0.212671 + g * 0.715160 + b * 0.072169;
        Dtype Z = r * 0.019334 + g * 0.119193 + b * 0.950227;

        out[src_idx+0*chan_stride] = X;
        out[src_idx+1*chan_stride] = Y;
        out[src_idx+2*chan_stride] = Z;
    }
}

template <typename Dtype>
__global__ void RGB2XYZBackward(const int n, const Dtype* top_diff, Dtype* bottom_diff, const Dtype* bottom_data,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width);
        int px = index % (height*width);
        int src_idx = blob_idx * height*width*chans // blob n index
            + px; // pixel index
        int chan_stride = height*width;

        Dtype dx = top_diff[src_idx + 0*chan_stride];
        Dtype dy = top_diff[src_idx + 1*chan_stride];
        Dtype dz = top_diff[src_idx + 2*chan_stride];

        Dtype drp = 0.412453*dx + 0.212671*dy +  0.019334*dz;
        Dtype dgp = 0.357580*dx + 0.715160*dy +  0.119193*dz;
        Dtype dbp = 0.180423*dx + 0.072169*dy +  0.950227*dz;

        Dtype r = bottom_data[src_idx + 0*chan_stride];
        Dtype g = bottom_data[src_idx + 1*chan_stride];
        Dtype b = bottom_data[src_idx + 2*chan_stride];

        Dtype dr = Dtype(0);
        Dtype dg = Dtype(0);
        Dtype db = Dtype(0);

        if ( r > 0.04045 ) {
            dr = 2.4/1.055*pow(( r + 0.055 ) / 1.055 , 1.4);
        } else {
            dr = 1.0 / 12.92;
        }  
        if ( g > 0.04045 ) {
            dg = 2.4/1.055*pow(( g + 0.055 ) / 1.055 , 1.4);
        } else {
            dg = 1.0 / 12.92;
        }  
        if ( b > 0.04045 ) {
            db = 2.4/1.055*pow(( b + 0.055 ) / 1.055 , 1.4);
        } else {
            db = 1.0 / 12.92;
        }  

        bottom_diff[src_idx+0*chan_stride] = dr*drp;
        bottom_diff[src_idx+1*chan_stride] = dg*dgp;
        bottom_diff[src_idx+2*chan_stride] = db*dbp;
    }
}

template <typename Dtype>
void RGB2XYZLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[2]*shape[3];
    RGB2XYZForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void RGB2XYZLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff    = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff       = bottom[0]->mutable_cpu_diff();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[2]*shape[3];
    RGB2XYZBackward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, top_diff, bottom_diff, bottom_data, 
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RGB2XYZLayer);

}  // namespace caffe
