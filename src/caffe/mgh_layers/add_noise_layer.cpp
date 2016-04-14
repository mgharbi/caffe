#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/add_noise_layer.hpp"

namespace caffe {

template <typename Dtype>
void AddNoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    for(size_t i = 1; i < top.size(); ++i) {
      CHECK_NE(top[i], bottom[0]) << this->type() << " Layer allows "
        "in-place computation only for the first output blob.";
    }

    noise_level_.clear();
    const AddNoiseParameter& add_noise_param = this->layer_param_.add_noise_param();
    std::copy(add_noise_param.noise_level().begin(),
            add_noise_param.noise_level().end(),
            std::back_inserter(noise_level_));
    CHECK_GE(noise_level_.size(), 1) << "AddNoiseLayer should have at least one noise_level ";
}

template <typename Dtype>
void AddNoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  if(top.size() > 1) {
      vector<int> shape = bottom[0]->shape();
      shape[1] = 1;
      shape[2] = 1;
      shape[3] = 1;
      top[1]->Reshape(shape);
  }
  if(top.size() > 2) {
      vector<int> shape = bottom[0]->shape();
      shape[1] = noise_level_.size();
      shape[2] = 1;
      shape[3] = 1;
      top[2]->Reshape(shape);
  }
}

template <typename Dtype>
void AddNoiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data          = top[0]->mutable_cpu_data();

    // top 1 is the noise level
    // top 2 is a 0-1 vector indicating which level has been chosen

    int n_noise_levels = noise_level_.size();

    vector<int> shape = bottom[0]->shape();

    vector<int> imshape = bottom[0]->shape();
    imshape[0] = 1;
    
    for (int n = 0; n < shape[0]; ++n) {
        // Sample a noise level
        int select = caffe_rng_rand() % noise_level_.size();
        float noise_std = noise_level_[select];

        Blob<Dtype> noise;
        noise.Reshape(imshape);
        int count = noise.count();

        Dtype *noise_data = noise.mutable_cpu_data();
        caffe_rng_gaussian(count, Dtype(0), Dtype(noise_std), noise_data );

        caffe_add(count, bottom_data+bottom[0]->offset(n), noise_data, top_data+top[0]->offset(n));

        if(top.size() > 1) {
            caffe_set(1, Dtype(noise_std), top[1]->mutable_cpu_data()+top[1]->offset(n));
        }
        if(top.size() > 2) {
            Dtype* noise_select = top[2]->mutable_cpu_data();
            caffe_set(n_noise_levels, Dtype(0), noise_select+top[2]->offset(n));
            caffe_set(1, Dtype(1), noise_select+top[2]->offset(n,select));
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(AddNoiseLayer);
#endif

INSTANTIATE_CLASS(AddNoiseLayer);
REGISTER_LAYER_CLASS(AddNoise);

}  // namespace caffe
