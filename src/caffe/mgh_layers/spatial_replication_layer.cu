#include <vector>

#include "caffe/mgh_layers/spatial_replication_layer.hpp"

namespace caffe {


template <typename Dtype>
void SpatialReplicationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) 
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int count = top[0]->height()*top[0]->width();
    for (int n = 0; n < top[0]->num(); ++n)
    for (int z = 0; z < top[0]->channels(); ++z)
    {
        caffe_gpu_set(count, bottom_data[bottom[0]->offset(n,z,0,0)], top_data+top[0]->offset(n,z));
    }
}

template <typename Dtype>
void SpatialReplicationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}


INSTANTIATE_LAYER_GPU_FUNCS(SpatialReplicationLayer);

}  // namespace caffe
