#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/flipLR_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FlipLRForward(const int n, const Dtype* in, Dtype* out, const float * randomize,
    int chans, int height, int width)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width*chans);

        int vx = index % (height*width*chans);
        int z = vx / (height*width);
        int px = vx % (height*width);
        int y = px/width;
        int x = px % width;

        int dst_idx = x + width*(y + height * (z + chans*blob_idx));

        // Flip only half of the images on average
        float f = randomize[0];
        int src_idx = width-1-x + width*(y + height * (z + chans*blob_idx));
        out[dst_idx] = f > 0.5f ? in[dst_idx] : in[src_idx];
    }
}

template <typename Dtype>
void FlipLRLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int w = shape[3];

    vector<int> rand_shape(1);
    rand_shape[0] = shape[0];
    Blob<float> randomize(rand_shape);
    caffe_rng_uniform<float>(shape[0], 0.0, 1.0, randomize.mutable_cpu_data());

    LOG(INFO) << "randomize " << randomize.cpu_data()[0];

    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    FlipLRForward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, randomize.gpu_data() ,
            shape[1], shape[2], shape[3]);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // delete[] randomize;
}

INSTANTIATE_LAYER_GPU_FUNCS(FlipLRLayer);

}  // namespace caffe
