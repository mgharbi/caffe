#include <vector>

#include "caffe/filler.hpp"
#include "caffe/mgh_layers/synth_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void SynthDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) 
{

    const int num_top = top.size();
    const SynthDataParameter& param = this->layer_param_.synth_data_param();

    CHECK(param.shape_size() == 1)
        << "Must specify 'shape' once"
        << "(" << num_top << "); specified " << param.shape_size() << ".";

    CHECK(param.shape(0).dim(1) == 3)
        << " SynthDataLayer generates 3 channel images";

    top[0]->Reshape(param.shape(0));
}

template <typename Dtype>
void SynthDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{

    Dtype* top_data = top[0]->mutable_cpu_data();
    vector<int> shape = top[0]->shape();

    // shape holds (n,c,h,w):
    // - n patches per batch
    // - c channels per patch
    // - h height
    // - w width
    for (int n = 0; n < shape[0]; ++n) { // each sample 
        // TODO: generate a patch for each patch

        // This is how you access pixel, x,y, channel c of patch n
        // top_data[top[0]->offset(n,c,y,x)] = 0.0;
    }
}

INSTANTIATE_CLASS(SynthDataLayer);
REGISTER_LAYER_CLASS(SynthData);

}  // namespace caffe
