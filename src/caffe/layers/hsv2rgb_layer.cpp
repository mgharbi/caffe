#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void HSV2RGBLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to rgb2hsv layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void HSV2RGBLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HSV2RGBLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        Dtype h = bottom_data[bottom[0]->offset(n,0,y,x)];
        Dtype s = bottom_data[bottom[0]->offset(n,1,y,x)];
        Dtype v = bottom_data[bottom[0]->offset(n,2,y,x)];

        int hi = std::floor(h*6);
        Dtype f = h*6-hi;
        Dtype p = v*(1-s);
        Dtype q = v*(1-f*s);
        Dtype t = v*(1-(1-f)*s);

        Dtype r = 0;
        Dtype g = 0;
        Dtype b = 0;
        switch(hi%6) {
            case 0:
                r = v;
                g = t;
                b = p;
                break;
            case 1:
                r = q;
                g = v;
                b = p;
                break;
            case 2:
                r = p;
                g = v;
                b = t;
                break;
            case 3:
                r = p;
                g = q;
                b = v;
                break;
            case 4:
                r = t;
                g = p;
                b = v;
                break;
            case 5:
                r = v;
                g = p;
                b = q;
                break;
        }

        top_data[top[0]->offset(n,0,y,x)] = r;
        top_data[top[0]->offset(n,1,y,x)] = g;
        top_data[top[0]->offset(n,2,y,x)] = b;
    }
}

#ifdef CPU_ONLY
STUB_GPU(HSV2RGBLayer);
#endif

INSTANTIATE_CLASS(HSV2RGBLayer);
REGISTER_LAYER_CLASS(HSV2RGB);

}  // namespace caffe
