#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/rot90_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Rot90Forward(const int n, const Dtype* in, Dtype* out, const float * randomize,
    int chans, int height, int width)
{
    CUDA_KERNEL_LOOP(index, n) {
        int blob_idx = index / (height*width*chans);

        int vx = index % (height*width*chans);
        int z  = vx / (height*width);
        int px = vx % (height*width);
        int y  = px/width;
        int x  = px % width;

        int xp = x;
        int yp = y;

        if(randomize[blob_idx] < 0.25f) { // angle 0
            xp = x;
            yp = y;
        } else if(randomize[blob_idx] < 0.5f) { // angle pi/2
            xp = height-1-y;
            yp = x;
        } else if(randomize[blob_idx] < 0.75f) { // angle pi
            xp = width-1-x;
            yp = height-1-y;
        } else { // angle 3*pi/2
            xp = width-1-x;
            yp = y;
        }
        int src_idx = xp + width*(yp + height * (z + chans*blob_idx));

        // Flip only half of the images on average
        int dst_idx = x + width*(y + height * (z + chans*blob_idx));
        out[dst_idx] = in[src_idx];
    }
}

template <typename Dtype>
void Rot90Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    vector<int> shape = bottom[0]->shape();
    int w = shape[3];

    vector<int> rand_shape(1);
    rand_shape[0] = shape[0];
    Blob<float> randomize(rand_shape);
    caffe_rng_uniform<float>(shape[0], 0.0, 1.0, randomize.mutable_cpu_data());

    int npix = shape[0]*shape[1]*shape[2]*shape[3];
    Rot90Forward<Dtype><<<CAFFE_GET_BLOCKS(npix), CAFFE_CUDA_NUM_THREADS>>>(
            npix, bottom_data, top_data, randomize.gpu_data() ,
            shape[1], shape[2], shape[3]);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(Rot90Layer);

}  // namespace caffe
