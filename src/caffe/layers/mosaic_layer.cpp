#include <cfloat>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void MosaicLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape()[1], 3) << "Input to mosaic layer should have 3 channels" ;
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void MosaicLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[1] = 1;
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
            if ( x % 2 == 0) {
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,1,y,x)];
            } else {
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,0,y,x)];
            }
        } else {
            if ( x % 2 == 0) {
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,2,y,x)];
            } else {
                top_data[top[0]->offset(n,0,y,x)] = bottom_data[bottom[0]->offset(n,1,y,x)];
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
