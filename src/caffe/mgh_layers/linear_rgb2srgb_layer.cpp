#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/linear_rgb2srgb_layer.hpp"

namespace caffe {

template <typename Dtype>
void LinearRGB2SRGBLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to srgb2linear_rgb layer should have 3 channels" ;
}

template <typename Dtype>
void LinearRGB2SRGBLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LinearRGB2SRGBLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    const float t_ = 0.0031308;
    const float a_ = 0.055;
    const float b_ = 1+a_;
    const float c_ = 1.0/2.4;
    const float d_ = 12.92;
    for (int x = 0; x < bottom[0]->count(); ++x) 
    {
        Dtype px = bottom_data[x];

        // inverse sRGB companding
        if ( px > t_ ) {
            px = b_*pow(px,c_) - a_;
        } else {
            px = px * d_;
        }  
        top_data[x] = px;
    }
}

template <typename Dtype>
void LinearRGB2SRGBLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

    const float t_ = 0.0031308;
    const float a_ = 0.055;
    const float b_ = 1+a_;
    const float c_ = 1.0/2.4;
    const float d_ = 12.92;
    for (int x = 0; x < bottom[0]->count(); ++x) 
    {
        Dtype dx = top_diff[x];
        Dtype px = bottom_data[x];
        Dtype dr = Dtype(0);

        if ( px > t_ ) {
            dr = b_*c_*pow(px,c_-1) ;
        } else {
            dr = d_;
        }  

        bottom_diff[x] = dr*dx;
    }
}

#ifdef CPU_ONLY
STUB_GPU(LinearRGB2SRGBLayer);
#endif

INSTANTIATE_CLASS(LinearRGB2SRGBLayer);
REGISTER_LAYER_CLASS(LinearRGB2SRGB);

}  // namespace caffe
