#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/add_scalar_layer.hpp"

namespace caffe {

template <typename Dtype>
void AddScalarLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  value_ = Dtype(this->layer_param_.add_scalar_param().value());
}

template <typename Dtype>
void AddScalarLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void AddScalarLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int count = bottom[0]->count();

    if(top[0] != bottom[0]) {
        caffe_copy(count, bottom_data, top_data);
    }
    caffe_add_scalar(count, value_, top_data);
}

template <typename Dtype>
void AddScalarLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int count = bottom[0]->count();
    if(top[0] != bottom[0]) {
        caffe_copy(count, top_diff, bottom_diff);
    }
}

#ifdef CPU_ONLY
STUB_GPU(AddScalarLayer);
#endif

INSTANTIATE_CLASS(AddScalarLayer);
REGISTER_LAYER_CLASS(AddScalar);

}  // namespace caffe
