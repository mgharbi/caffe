#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/xyz2lab_normalized_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void XYZ2LABNormalizedForward(const int n, const Dtype* in, Dtype* out,
    double X_ref, double Y_ref, double Z_ref,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width);
        int px = index % (height*width);
        int src_idx = blob_idx * height*width*chans // blob n index
            + px; // pixel index
        int chan_stride = height*width;

        Dtype X = in[src_idx + 0*chan_stride];
        Dtype Y = in[src_idx + 1*chan_stride];
        Dtype Z = in[src_idx + 2*chan_stride];

        X  = X/X_ref;
        Y  = Y/Y_ref;
        Z  = Z/Z_ref;

        if ( X > 0.008856 ){
            X = pow(Dtype(X),  Dtype(1.0/3.0) );
        } else {
            X = ( 7.787 * X ) + ( 16.0 / 116.0 );
        }   
        if ( Y > 0.008856 ){
            Y = pow(Dtype(Y),  Dtype(1.0/3.0) );
        } else {
            Y = ( 7.787 * Y ) + ( 16.0 / 116.0 );
        }   
        if ( Z > 0.008856 ){
            Z = pow(Dtype(Z),  Dtype(1.0/3.0) );
        } else {
            Z = ( 7.787 * Z ) + ( 16.0 / 116.0 );
        }   

        double L = 116.0*Y - 16.0;
        double a = 500.0*(X-Y);
        double b = 200.0*(Y-Z);

        out[src_idx+0*chan_stride] = L/100.0;
        out[src_idx+1*chan_stride] = (a+127.0)/255.0;
        out[src_idx+2*chan_stride] = (b+127.0)/255.0;
    }
}

template <typename Dtype>
__global__ void XYZ2LABNormalizedBackward(const int n, const Dtype* top_diff, Dtype* bottom_diff, const Dtype* bottom_data,
    double X_ref, double Y_ref, double Z_ref,
    int height, int width, int chans)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width);
        int px = index % (height*width);
        int src_idx = blob_idx * height*width*chans // blob n index
            + px; // pixel index
        int chan_stride = height*width;

        Dtype dL = top_diff[src_idx + 0*chan_stride]/100.0;
        Dtype da = top_diff[src_idx + 1*chan_stride]/255.0;
        Dtype db = top_diff[src_idx + 2*chan_stride]/255.0;

        Dtype dX_pp =            500.0*da           ;
        Dtype dY_pp = 116.0*dL - 500.0*da + 200.0*db;
        Dtype dZ_pp =                     - 200.0*db;

        Dtype X = bottom_data[src_idx + 0*chan_stride];
        Dtype Y = bottom_data[src_idx + 1*chan_stride];
        Dtype Z = bottom_data[src_idx + 2*chan_stride];

        Dtype Xp = X / X_ref;
        Dtype Yp = Y / Y_ref;
        Dtype Zp = Z / Z_ref;

        Dtype dX_p = 0;
        Dtype dY_p = 0;
        Dtype dZ_p = 0;

        if ( Xp > 0.008856 ){
            dX_p = 1.0/3.0 * pow(Dtype(Xp),  Dtype(-2.0/3.0) );
        } else {
            dX_p = 7.787;
        }   
        if ( Yp > 0.008856 ){
            dY_p = 1.0/3.0 * pow(Dtype(Yp),  Dtype(-2.0/3.0) );
        } else {
            dY_p = 7.787;
        }   
        if ( Zp > 0.008856 ){
            dZ_p = 1.0/3.0 * pow(Dtype(Zp),  Dtype(-2.0/3.0) );
        } else {
            dZ_p = 7.787;
        }   

        bottom_diff[src_idx+0*chan_stride] = dX_p*dX_pp/X_ref;
        bottom_diff[src_idx+1*chan_stride] = dY_p*dY_pp/Y_ref;
        bottom_diff[src_idx+2*chan_stride] = dZ_p*dZ_pp/Z_ref;
    }
}

template <typename Dtype>
void XYZ2LABNormalizedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[2]*shape[3];
    XYZ2LABNormalizedForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            X_ref, Y_ref, Z_ref,
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void XYZ2LABNormalizedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff    = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff       = bottom[0]->mutable_cpu_diff();

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[2]*shape[3];
    XYZ2LABNormalizedBackward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, top_diff, bottom_diff, bottom_data, 
            X_ref, Y_ref, Z_ref,
            shape[3], shape[2],shape[1]);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(XYZ2LABNormalizedLayer);

}  // namespace caffe
