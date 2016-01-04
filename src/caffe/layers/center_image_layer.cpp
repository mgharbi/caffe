#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void CenterImageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void CenterImageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CenterImageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int count = bottom[0]->count();

    caffe_copy(count, bottom_data, top_data);
    caffe_add_scalar(count, Dtype(-0.5), top_data);
}

template <typename Dtype>
void CenterImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int count = bottom[0]->count();
    caffe_copy(count, top_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(CenterImageLayer);
#endif

INSTANTIATE_CLASS(CenterImageLayer);
REGISTER_LAYER_CLASS(CenterImage);

}  // namespace caffe
