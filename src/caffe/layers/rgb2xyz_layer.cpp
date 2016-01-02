#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void RGB2XYZLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to rgb2xyz layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void RGB2XYZLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RGB2XYZLayer<Dtype>::Forward_cpu(
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

        if ( r > 0.04045 ) {
            r = pow(( r + 0.055 ) / 1.055 , 2.4);
        } else {
            r = r / 12.92;
        }  
        if ( g > 0.04045 ) {
            g = pow(( g + 0.055 ) / 1.055 , 2.4);
        } else {
            g = g / 12.92;
        }  
        if ( b > 0.04045 ) {
            b = pow(( b + 0.055 ) / 1.055 , 2.4);
        } else {
            b = b / 12.92;
        }  

        //Observer. = 2°, Illuminant = D65
        Dtype X = r * 0.412453 + g * 0.357580 + b * 0.180423;
        Dtype Y = r * 0.212671 + g * 0.715160 + b * 0.072169;
        Dtype Z = r * 0.019334 + g * 0.119193 + b * 0.950227;

        top_data[top[0]->offset(n,0,y,x)] = X;
        top_data[top[0]->offset(n,1,y,x)] = Y;
        top_data[top[0]->offset(n,2,y,x)] = Z;
    }
}

template <typename Dtype>
void RGB2XYZLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        Dtype dx = top_diff[top[0]->offset(n,0,y,x)];
        Dtype dy = top_diff[top[0]->offset(n,1,y,x)];
        Dtype dz = top_diff[top[0]->offset(n,2,y,x)];

        Dtype drp = 0.412453*dx + 0.212671*dy +  0.019334*dz;
        Dtype dgp = 0.357580*dx + 0.715160*dy +  0.119193*dz;
        Dtype dbp = 0.180423*dx + 0.072169*dy +  0.950227*dz;

        Dtype r = bottom_data[bottom[0]->offset(n,0,y,x)];
        Dtype g = bottom_data[bottom[0]->offset(n,1,y,x)];
        Dtype b = bottom_data[bottom[0]->offset(n,2,y,x)];

        Dtype dr = Dtype(0);
        Dtype dg = Dtype(0);
        Dtype db = Dtype(0);

        if ( r > 0.04045 ) {
            dr = 2.4/1.055*pow(( r + 0.055 ) / 1.055 , 1.4);
        } else {
            dr = 1.0 / 12.92;
        }  
        if ( g > 0.04045 ) {
            dg = 2.4/1.055*pow(( g + 0.055 ) / 1.055 , 1.4);
        } else {
            dg = 1.0 / 12.92;
        }  
        if ( b > 0.04045 ) {
            db = 2.4/1.055*pow(( b + 0.055 ) / 1.055 , 1.4);
        } else {
            db = 1.0 / 12.92;
        }  

        bottom_diff[bottom[0]->offset(n,0,y,x)] = dr*drp;
        bottom_diff[bottom[0]->offset(n,1,y,x)] = dg*dgp;
        bottom_diff[bottom[0]->offset(n,2,y,x)] = db*dbp;
    }
}

#ifdef CPU_ONLY
STUB_GPU(RGB2XYZLayer);
#endif

INSTANTIATE_CLASS(RGB2XYZLayer);
REGISTER_LAYER_CLASS(RGB2XYZ);

}  // namespace caffe
