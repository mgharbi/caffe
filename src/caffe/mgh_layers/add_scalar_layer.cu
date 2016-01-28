#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/add_scalar_layer.hpp"

namespace caffe {


template <typename Dtype>
void AddScalarLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int count = bottom[0]->count();
    if(top[0] != bottom[0]) {
        caffe_copy(count, bottom_data, top_data);
    }

    caffe_gpu_add_scalar(count, value_, top_data);
}

template <typename Dtype>
void AddScalarLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int count = bottom[0]->count();

    if(top[0] != bottom[0]) {
        caffe_copy(count, top_diff, bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(AddScalarLayer);
}  // namespace caffe
