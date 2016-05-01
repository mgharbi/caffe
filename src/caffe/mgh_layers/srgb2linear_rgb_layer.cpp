#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/srgb2linear_rgb_layer.hpp"

namespace caffe {

template <typename Dtype>
void SRGB2LinearRGBLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SRGB2LinearRGBLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SRGB2LinearRGBLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for (int x = 0; x < bottom[0]->count(); ++x) 
    {
        Dtype px = bottom_data[x];

        // inverse sRGB companding
        if ( px > 0.04045 ) {
            px = pow(( px + 0.055 ) / 1.055 , 2.4);
        } else {
            px = px / 12.92;
        }  
        top_data[x] = px;
    }
}

template <typename Dtype>
void SRGB2LinearRGBLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // Loss L
    // y = f(x)

    // Backpropagation:
    //    - compute dL/dx as a function of dL/dy
    //    - chain rule dL/dx = dL/dy* dy/dx
    //    - top_diff is dL/dy

    for (int x = 0; x < bottom[0]->count(); ++x) 
    {
        Dtype dx = top_diff[x];
        Dtype px = bottom_data[x];
        Dtype dr = Dtype(0);

        if ( px > 0.04045) {
            dr = 2.4/1.055*pow(( px + 0.055 ) / 1.055  , 2.4-1);
        } else {
            dr = 1.0/12.92;
        }  

        bottom_diff[x] = dr*dx;
    }
}

#ifdef CPU_ONLY
STUB_GPU(SRGB2LinearRGBLayer);
#endif

INSTANTIATE_CLASS(SRGB2LinearRGBLayer);
REGISTER_LAYER_CLASS(SRGB2LinearRGB);

}  // namespace caffe
