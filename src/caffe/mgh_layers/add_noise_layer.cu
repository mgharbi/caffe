#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/add_noise_layer.hpp"

namespace caffe {


template <typename Dtype>
void AddNoiseLayer<Dtype>::Forward_gpu(
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data          = top[0]->mutable_gpu_data();

    // top 1 is the noise level
    // top 2 is a 0-1 vector indicating which level has been chosen

    int n_noise_levels = noise_level_.size();

    vector<int> shape = bottom[0]->shape();

    vector<int> imshape = bottom[0]->shape();
    imshape[0] = 1;
    
    if(mode_ == AddNoiseParameter_NoiseMode_DISCRETE) {
        for (int n = 0; n < shape[0]; ++n) {
            // Sample a noise level
            int select = caffe_rng_rand() % noise_level_.size();
            float noise_std = noise_level_[select];

            Blob<Dtype> noise;
            noise.Reshape(imshape);
            int count = noise.count();

            if(noise_std > 0) {
                Dtype *noise_data = noise.mutable_gpu_data();
                caffe_gpu_rng_gaussian(count, Dtype(0), Dtype(noise_std), noise_data );
                caffe_gpu_add(count, bottom_data+bottom[0]->offset(n), noise_data, top_data+top[0]->offset(n));
            }else {
                caffe_copy(count, bottom_data+bottom[0]->offset(n), top_data+top[0]->offset(n));
            }

            if(top.size() > 1) {
                caffe_gpu_set(1, Dtype(noise_std), top[1]->mutable_gpu_data()+top[1]->offset(n));
            }
            if(top.size() > 2) {
                Dtype* noise_select = top[2]->mutable_gpu_data();
                caffe_gpu_set(n_noise_levels, Dtype(0), noise_select+top[2]->offset(n));
                caffe_gpu_set(1, Dtype(1), noise_select+top[2]->offset(n,select));
            }
        }
    } else {
        // Sample a few noise levels
        float *noise_std = new float[shape[0]];
        caffe_rng_uniform(shape[0], noise_level_[0], noise_level_[1], noise_std);
        for (int n = 0; n < shape[0]; ++n) {
            Blob<Dtype> noise;
            noise.Reshape(imshape);
            int count = noise.count();

            if(noise_std > 0) {
                Dtype *noise_data = noise.mutable_gpu_data();
                caffe_gpu_rng_gaussian(count, Dtype(0), Dtype(noise_std[n]), noise_data );
                caffe_gpu_add(count, bottom_data+bottom[0]->offset(n), noise_data, top_data+top[0]->offset(n));
            }else {
                caffe_copy(count, bottom_data+bottom[0]->offset(n), top_data+top[0]->offset(n));
            }

            if(top.size() > 1) {
                caffe_gpu_set(1, Dtype(noise_std[n]), top[1]->mutable_gpu_data()+top[1]->offset(n));
            }
        }
        delete[] noise_std;
    }
}

template <typename Dtype>
void AddNoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(AddNoiseLayer);
}  // namespace caffe
