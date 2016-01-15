#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers/mosaic_layer.hpp"

namespace caffe {

template <typename Dtype>
void MosaicLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to mosaic layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  store_pattern_ = this->layer_param_.mosaic_param().store_pattern();
}

template <typename Dtype>
void MosaicLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  if(store_pattern_) {
      shape[1] = 4;
  } else {
      shape[1] = 1;
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MosaicLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    // Mosaick
    // G R G R G
    // B G B G B
    // G R G R G

    vector<int> shape = bottom[0]->shape();
    for (int n = 0; n < shape[0]; ++n) 
    for (int y = 0; y < shape[2]; ++y) 
    for (int x = 0; x < shape[3]; ++x) 
    {
        if(y % 2 == 0) {
            if ( x % 2 == 0) { // G
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,1,y,x)];
                if(store_pattern_){
                    top_data[top[0]->offset(n,2,y,x)] = 1;
                }
            } else { // R
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,0,y,x)];
                if(store_pattern_){
                    top_data[top[0]->offset(n,1,y,x)] = 1;
                }
            }
        } else {
            if ( x % 2 == 0) { // B
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,2,y,x)];
                if(store_pattern_){
                    top_data[top[0]->offset(n,3,y,x)] = 1;
                }
            } else { // G
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,1,y,x)];
                if(store_pattern_){
                    top_data[top[0]->offset(n,2,y,x)] = 1;
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(MosaicLayer);
#endif

INSTANTIATE_CLASS(MosaicLayer);
REGISTER_LAYER_CLASS(Mosaic);

}  // namespace caffe
