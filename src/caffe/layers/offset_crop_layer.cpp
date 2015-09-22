#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/mgh_layers.hpp"

namespace caffe {

template <typename Dtype>
void OffsetCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int crop_h_ = (bottom[0]->height()-bottom[1]->height());
  int crop_w_ = (bottom[0]->width()-bottom[1]->width());
  CHECK_GE(crop_h_,0) << "Crop size must be stricly positive";
  CHECK_GE(crop_w_,0) << "Crop size must be stricly positive";

  int index = this->layer_param_.offset_crop_param().index();
  CHECK_LE(index,bottom[2]->width()) << "Crop must be smaller thant target";

  offset_x = bottom[2]->data_at(0, 0, 0, index);
  offset_y = bottom[2]->data_at(0, 0, 1, index);
  CHECK_GE(offset_x,0) << "Offset must be positive";
  CHECK_GE(offset_y,0) << "Offset must be positive";
  CHECK_LE(offset_x+bottom[1]->width(),bottom[0]->width()) << "Crop must be smaller thant target";
  CHECK_GE(offset_y+bottom[0]->height(),bottom[0]->height()) << "Crop must be smaller thant target";
}

template <typename Dtype>
void OffsetCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(),
            bottom[1]->width());
}

template <typename Dtype>
void OffsetCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int n = 0; n < top[0]->num(); ++n) {
        for (int c = 0; c < top[0]->channels(); ++c) {
            for (int h = 0; h < top[0]->height(); ++h) {
                caffe_copy(top[0]->width(),
                        bottom_data + bottom[0]->offset(n, c, offset_y + h, offset_x),
                        top_data + top[0]->offset(n, c, h));
            }
        }
    }
}

template <typename Dtype>
void OffsetCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, offset_y + h, offset_x));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(OffsetCropLayer);
#endif

INSTANTIATE_CLASS(OffsetCropLayer);
REGISTER_LAYER_CLASS(OffsetCrop);

}  // namespace caffe
