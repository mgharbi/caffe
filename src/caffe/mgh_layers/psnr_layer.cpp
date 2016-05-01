#include <vector>

#include "caffe/mgh_layers/psnr_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PSNRLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PSNRLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), 1)
      << "Input should be a singleton MSE value";
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void PSNRLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype psnr = -10*log10(bottom[0]->cpu_data()[0] * Dtype(2));
  top[0]->mutable_cpu_data()[0] = psnr;

}

#ifdef CPU_ONLY
STUB_GPU(PSNRLayer);
#endif

INSTANTIATE_CLASS(PSNRLayer);
REGISTER_LAYER_CLASS(PSNR);

}  // namespace caffe
