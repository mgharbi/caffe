#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/mosaic_offset_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

#include <math_functions.hpp>

namespace caffe {

template <typename Dtype>
__global__ void MosaicOffsetForward(const int n, const Dtype* in, Dtype* out,
    int height, int width, int chans, const int* offsets)
{
    int offset_x = offsets[0];
    int offset_y = offsets[1];
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width*chans);

        int vx = index % (height*width*chans);
        int z  = vx / (height*width);
        int px = vx % (height*width);
        int y  = px/width;
        int x  = px % width;

        int dst_idx = x + width*(y + height * (z + chans*blob_idx));
        int src_idx = max(x-offset_x,0) + width*(max(y-offset_y,0) + height * (z + chans*blob_idx));

        out[dst_idx] = in[src_idx];
    }
}

template <typename Dtype>
void MosaicOffsetLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = top[0]->count();

    // Process only during training
    if (this->phase_ != TRAIN) {
        caffe_copy(count, bottom_data, top_data);
        return;
    }

    int offset_x = Rand(2);
    int offset_y = Rand(2);

    vector<int> ofshape(1); ofshape[0] = 2;
    Blob<int> offsets(ofshape);
    offsets.mutable_cpu_data()[0] = offset_x;
    offsets.mutable_cpu_data()[1] = offset_y;

    // No offset, copy data over
    if(offset_x == 0 && offset_y == 0) {
        caffe_copy(count, bottom_data, top_data);
        return;
    }

    vector<int> shape = bottom[0]->shape();
    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    MosaicOffsetForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, 
            shape[3], shape[2],shape[1], offsets.gpu_data());

}
template <typename Dtype>
void MosaicOffsetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}
INSTANTIATE_LAYER_GPU_FUNCS(MosaicOffsetLayer);

}  // namespace caffe
