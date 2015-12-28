#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void XYZ2LABLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  X_ref = .95047;
  Y_ref = 1.;
  Z_ref = 1.08883;

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to xyz2lab layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void XYZ2LABLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void XYZ2LABLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();


    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {

        Dtype X = bottom_data[bottom[0]->offset(n,0,y,x)];
        Dtype Y = bottom_data[bottom[0]->offset(n,1,y,x)];
        Dtype Z = bottom_data[bottom[0]->offset(n,2,y,x)];

        X /= X_ref;
        Y /= Y_ref;
        Z /= Z_ref;

        if ( X > 0.008856 ){
            X = pow(X,  1.0/3.0 );
        } else {
            X = ( 7.787 * X ) + ( 16.0 / 116.0 );
        }   
        if ( Y > 0.008856 ){
            Y = pow(Y,  1.0/3.0 );
        } else {
            Y = ( 7.787 * Y ) + ( 16.0 / 116.0 );
        }   
        if ( Z > 0.008856 ){
            Z = pow(Z,  1.0/3.0 );
        } else {
            Z = ( 7.787 * Z ) + ( 16.0 / 116.0 );
        }   

        float L = 116.0*Y - 16.0;
        float a = 500.0*(X-Y);
        float b = 200.0*(Y-Z);

        top_data[top[0]->offset(n,0,y,x)] = L;
        top_data[top[0]->offset(n,1,y,x)] = a;
        top_data[top[0]->offset(n,2,y,x)] = b;
    }
}

template <typename Dtype>
void XYZ2LABLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = top[0]->mutable_cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        Dtype dL = top_diff[top[0]->offset(n,0,y,x)];
        Dtype da = top_diff[top[0]->offset(n,1,y,x)];
        Dtype db = top_diff[top[0]->offset(n,2,y,x)];

        Dtype dX_pp =            500.0*da            ;
        Dtype dY_pp = 116.0*dL - 500.0*da +  200.0*db;
        Dtype dZ_pp =                        200.0*db;

        Dtype X = bottom_data[bottom[0]->offset(n,0,y,x)];
        Dtype Y = bottom_data[bottom[0]->offset(n,1,y,x)];
        Dtype Z = bottom_data[bottom[0]->offset(n,2,y,x)];

        Dtype Xp = X / X_ref;
        Dtype Yp = Y / Y_ref;
        Dtype Zp = Z / Z_ref;

        Dtype dX_p = Dtype(0);
        Dtype dY_p = Dtype(0);
        Dtype dZ_p = Dtype(0);

        if ( Xp > 0.008856 ){
            dX_p = 1.0/3.0*pow(Xp,  -2.0/3.0 );
        } else {
            dX_p = 7.787;
        }   
        if ( Yp > 0.008856 ){
            dY_p = 1.0/3.0*pow(Yp,  -2.0/3.0 );
        } else {
            dY_p = 7.787;
        }   
        if ( Zp > 0.008856 ){
            dZ_p = 1.0/3.0*pow(Zp,  -2.0/3.0 );
        } else {
            dZ_p = 7.787;
        }   

        dX_p *= dX_pp;
        dY_p *= dY_pp;
        dZ_p *= dZ_pp;

        bottom_diff[top[0]->offset(n,0,y,x)] = dX_p*1.0/X_ref;
        bottom_diff[top[0]->offset(n,1,y,x)] = dY_p*1.0/Y_ref;
        bottom_diff[top[0]->offset(n,2,y,x)] = dZ_p*1.0/Z_ref;
    }
}

#ifdef CPU_ONLY
STUB_GPU(XYZ2LABLayer);
#endif

INSTANTIATE_CLASS(XYZ2LABLayer);
REGISTER_LAYER_CLASS(XYZ2LAB);

}  // namespace caffe
