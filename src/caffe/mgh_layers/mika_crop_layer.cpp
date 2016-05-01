#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/mgh_layers/mika_crop_layer.hpp"

namespace caffe {

template <typename Dtype>
void MikaCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    MikaCropParameter crop_param = this->layer_param_.mika_crop_param();
    mode_ = crop_param.mode();
    crop_h_ = 0;
    crop_w_ = 0;
    switch(mode_) {
        case MikaCropParameter_CropMode_MATCH_SECOND_INPUT:
            // Then we compute the crop size from blob 0 - blob 1
            CHECK_EQ((bottom[0]->height()-bottom[1]->height()) % 2,0) << "Size difference must be even, not implemented for odd sizes";
            CHECK_EQ((bottom[0]->width()-bottom[1]->width()) % 2,0) << "Size difference must be even, not implemented for odd sizes";
            // Blob 1 has the reference dimensions
            crop_h_ = (bottom[0]->height()-bottom[1]->height())/2;
            crop_w_ = (bottom[0]->width()-bottom[1]->width())/2;
            break;
        case MikaCropParameter_CropMode_FROM_PROTO_PARAM:
            if (crop_param.has_crop_h() || crop_param.has_crop_w()) {
                CHECK_EQ(false, crop_param.has_crop_size())
                    << "Either crop_size or crop_h/w should be specified; not both.";
                crop_h_ = crop_param.crop_h();
                crop_w_ = crop_param.crop_w();
            } else if (crop_param.has_crop_size()){
                CHECK_EQ(false, crop_param.has_crop_w())
                    << "Either crop_size or crop_h/w should be specified; not both.";
                CHECK_EQ(false, crop_param.has_crop_h())
                    << "Either crop_size or crop_h/w should be specified; not both.";
                crop_h_ = crop_param.crop_size();
                crop_w_ = crop_param.crop_size();
            }
            break;
        case MikaCropParameter_CropMode_FROM_BLOB_PARAM:
            CHECK_EQ(bottom.size(),2) << "In 'from_blob' mode, the crop layer needs two inputs";
            CHECK_EQ(false, crop_param.has_crop_w())
                << "In 'from_blob' mode, the crop layer cannot take dimensions";
            CHECK_EQ(false, crop_param.has_crop_h())
                << "In 'from_blob' mode, the crop layer cannot take dimensions";
            CHECK_EQ(false, crop_param.has_crop_size())
                << "In 'from_blob' mode, the crop layer cannot take dimensions";
            CHECK_EQ(bottom[1]->count(),2) << " In 'from_blob' mode, the 3rd input to the MikaCropLayer needs to have 2 elements";
            crop_w_ = bottom[1]->cpu_data()[0];
            crop_h_ = bottom[1]->cpu_data()[1];
            break;
        default:
            LOG(FATAL) << "crop mode unknown";
    }


}

template <typename Dtype>
void MikaCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{

    if(mode_ == MikaCropParameter_CropMode_FROM_BLOB_PARAM) {
        CHECK_GE(bottom[1]->cpu_data()[0],0) << "Blob crop size should be positive";
        CHECK_GE(bottom[1]->cpu_data()[1],0) << "Blob crop size should be positive";
        crop_w_ = (bottom[0]->width()-bottom[1]->cpu_data()[0])/2;
        crop_h_ = (bottom[0]->height()-bottom[1]->cpu_data()[1])/2;
    }

    CHECK_GE(crop_h_,0) << "Crop size must be positive";
    CHECK_GE(crop_w_,0) << "Crop size must be positive";
    CHECK_LE(2*crop_h_+1, bottom[0]->height()) << "Crop size must be smaller than input to be cropped";
    CHECK_LE(2*crop_w_+1, bottom[0]->width()) << "Crop size must be smaller than input to be cropped";

    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height()-2*crop_h_,
            bottom[0]->width()-2*crop_w_);
}

template <typename Dtype>
void MikaCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int n = 0; n < top[0]->num(); ++n)
    for (int c = 0; c < top[0]->channels(); ++c)
    for (int h = 0; h < top[0]->height(); ++h) 
    {
        // Copy row by row
        caffe_copy(top[0]->width(),
            bottom_data + bottom[0]->offset(n, c, crop_h_ + h, crop_w_),
            top_data + top[0]->offset(n, c, h));
    }
}

template <typename Dtype>
void MikaCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (propagate_down[0]) {
        caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
        for (int n = 0; n < top[0]->num(); ++n)
        for (int c = 0; c < top[0]->channels(); ++c)
        for (int h = 0; h < top[0]->height(); ++h) 
        {
            caffe_copy(top[0]->width(),
                top_diff + top[0]->offset(n, c, h),
                bottom_diff + bottom[0]->offset(n, c, crop_h_ + h, crop_w_));
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(MikaCropLayer);
#endif

INSTANTIATE_CLASS(MikaCropLayer);
REGISTER_LAYER_CLASS(MikaCrop);

}  // namespace caffe
