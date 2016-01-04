#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void RGB2HSVLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to rgb2hsv layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void RGB2HSVLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RGB2HSVLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        Dtype r = bottom_data[bottom[0]->offset(n,0,y,x)];
        Dtype g = bottom_data[bottom[0]->offset(n,1,y,x)];
        Dtype b = bottom_data[bottom[0]->offset(n,2,y,x)];

        Dtype maxi = fmax(r,fmax(g,b));
        Dtype mini = fmin(r,fmin(g,b));
        Dtype delta = maxi-mini;

        Dtype h = 0;
        Dtype v = 0; 
        Dtype s = 0; 

        // Value
        v = maxi;

        // Saturation
        if(v == 0 || delta == 0) {
            s = 0;
        }else{
            s = delta/v;
        }

        // Hue
        if(delta == 0){
            h = 0;
        }else {
            if(maxi == r) {
                h = (g-b)/delta;
            }else if(maxi == g) {
                h = 2.0 + (b-r)/delta;
            }else if(maxi == b) {
                h = 4.0 + (r-g)/delta;
            }
            h = fmod((h+6)/6.0,1.0);
        }

        top_data[top[0]->offset(n,0,y,x)] = h;
        top_data[top[0]->offset(n,1,y,x)] = s;
        top_data[top[0]->offset(n,2,y,x)] = v;
    }
}

#ifdef CPU_ONLY
STUB_GPU(RGB2HSVLayer);
#endif

INSTANTIATE_CLASS(RGB2HSVLayer);
REGISTER_LAYER_CLASS(RGB2HSV);

}  // namespace caffe
