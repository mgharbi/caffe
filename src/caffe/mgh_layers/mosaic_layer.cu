#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/mosaic_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void mosaick_row_kernel(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dst_outer_stride, const int dst_inner_stride,
    bool store_pattern,
    const Dtype* src, Dtype* dst) 
{

    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height);
        int y = index % (height);
        int src_start = blob_idx * src_outer_stride // blob n index
            + y * src_inner_stride; // line index
        int dst_start = blob_idx * dst_outer_stride // blob n index
            + y * dst_inner_stride; // line index

        int chan_stride = height*width;

        if(y % 2 == 0) {
            for (int x = 0; x < width; ++x) {
                if ( x % 2 == 0) { // G
                    // top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,1,y,x)];
                    dst[dst_start + x] = src[src_start + x + 1*chan_stride];
                    if(store_pattern){
                        // top_data[top[0]->offset(n,2,y,x)] = 1;
                        dst[dst_start + x + 2*chan_stride] = 1;
                    }
                } else { // R
                    // top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,0,y,x)];
                    dst[dst_start + x] = src[src_start + x + 0*chan_stride];
                    if(store_pattern){
                        // top_data[top[0]->offset(n,1,y,x)] = 1;
                        dst[dst_start + x + 1*chan_stride] = 1;
                    }
                }
            }
        } else {
            for (int x = 0; x < width; ++x) {
                if ( x % 2 == 0) { // B
                    // top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,2,y,x)];
                    dst[dst_start + x] = src[src_start + x + 2*chan_stride];
                    if(store_pattern){
                        // top_data[top[0]->offset(n,3,y,x)] = 1;
                        dst[dst_start + x + 3*chan_stride] = 1;
                    }
                } else { // G
                    // top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,1,y,x)];
                    dst[dst_start + x] = src[src_start + x + 1*chan_stride];
                    if(store_pattern){
                        // top_data[top[0]->offset(n,2,y,x)] = 1;
                        dst[dst_start + x + 2*chan_stride] = 1;
                    }
                }
            }
        }
    }
}

template <typename Dtype>
void MosaicLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    const int lines = shape[0]*shape[2]; // n blobs, h lines per blob

    const int src_outer_stride = shape[3]*shape[2]*shape[1]; // skip h*w*c between blobs
    const int src_inner_stride = shape[3]; // skip w pixels between lines

    vector<int> tshape = top[0]->shape();
    const int dst_outer_stride = tshape[3]*tshape[2]*tshape[1];
    const int dst_inner_stride = tshape[3];

    mosaick_row_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
            lines, top[0]->height(), top[0]->width(),
            src_outer_stride, src_inner_stride,
            dst_outer_stride, dst_inner_stride,
            store_pattern_,
            bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
    // Mosaick
    // G R G R G
    // B G B G B
    // G R G R G
}


INSTANTIATE_LAYER_GPU_FUNCS(MosaicLayer);

}  // namespace caffe

