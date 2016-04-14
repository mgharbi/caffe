#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/mgh_layers/spatial_replication_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialReplicationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->width(), 1) << "first input should be a singleton spatially";
    CHECK_EQ(bottom[0]->height(), 1) << "first input should be a singleton spatially";

}

template <typename Dtype>
void SpatialReplicationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(),
            bottom[1]->width());
}

template <typename Dtype>
void SpatialReplicationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int count = top[0]->height()*top[0]->width();
    for (int n = 0; n < top[0]->num(); ++n)
    for (int z = 0; z < top[0]->channels(); ++z)
    {
        caffe_set(count, bottom_data[bottom[0]->offset(n,z,0,0)], top_data+top[0]->offset(n,z));
    }
}


#ifdef CPU_ONLY
STUB_GPU(SpatialReplicationLayer);
#endif

INSTANTIATE_CLASS(SpatialReplicationLayer);
REGISTER_LAYER_CLASS(SpatialReplication);

}  // namespace caffe
