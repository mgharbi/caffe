#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizedDeconvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = this->stride_h_ * (this->height_ - 1) + this->kernel_h_
      - 2 * this->pad_h_;
  this->width_out_ = this->stride_w_ * (this->width_ - 1) + this->kernel_w_
      - 2 * this->pad_w_;
}

template <typename Dtype>
void NormalizedDeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    const Dtype* weight = this->blobs_[0]->cpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->backward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                    top_data + top[i]->offset(n));
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->cpu_data();
                this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
            }
            this->normalize_boundaries_cpu(top_data + top[i]->offset(n), top_data + top[i]->offset(n));
        }
    }
}

template <typename Dtype>
void NormalizedDeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      vector<int> shape(4);
      shape[0] = 1;
      shape[1] = top[i]->shape(1);
      shape[2] = top[i]->shape(2);
      shape[3] = top[i]->shape(3);
      // Blob<Dtype> normalized(shape);
      // Dtype * norm_diff = normalized.mutable_cpu_data();
      // Dtype *normalized = new Dtype[top[i]->offset(1)]();
      for (int n = 0; n < this->num_; ++n) {
          // this->normalize_boundaries_cpu(top_diff + top[i]->offset(n), norm_diff);
          this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
      // delete normalized;
    }

    if (this->param_propagate_down_[0] || propagate_down[i]) {
        vector<int> shape(4);
        shape[0] = 1;
        shape[1] = top[i]->shape(1);
        shape[2] = top[i]->shape(2);
        shape[3] = top[i]->shape(3);
        // Blob<Dtype> normalized(shape);
        // Dtype * norm_diff = normalized.mutable_cpu_data();
        for (int n = 0; n < this->num_; ++n) {
            // this->normalize_boundaries_cpu(top_diff + top[i]->offset(n), norm_diff);
            
            // Gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
                this->weight_cpu_gemm(top_diff + top[i]->offset(n),
                        bottom_data + bottom[i]->offset(n), weight_diff);
                // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
                // we might have just computed above.
            }
            if (propagate_down[i]) {
                this->forward_cpu_gemm(top_diff + top[i]->offset(n), weight,
                        bottom_diff + bottom[i]->offset(n),
                        this->param_propagate_down_[0]);
            } 
        }
    }
  }
}

template <typename Dtype>
void NormalizedDeconvolutionLayer<Dtype>::normalize_boundaries_cpu(const Dtype* input, Dtype* output) {
    int channels = this->conv_in_channels_;
    int height   = this->conv_in_height_;
    int width    = this->conv_in_width_;
    int pad_h    = this->pad_h_;
    int pad_w    = this->pad_w_;
    int stride_h = this->stride_h_;
    int stride_w = this->stride_w_;
    int patch_h  = this->kernel_h_;
    int patch_w  = this->kernel_w_;

    int *count = new int[height*width*channels]();
    int height_col   = (height + 2 * pad_h - patch_h) / stride_h + 1;
    int width_col    = (width + 2 * pad_w - patch_w) / stride_w + 1;
    int channels_col = channels * patch_h * patch_w;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % patch_w;
        int h_offset = (c / patch_w) % patch_h;
        int c_im     = c / patch_h / patch_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                    count[(c_im * height + h_pad) * width + w_pad] += 1;
                }
            }
        }
    }
    for (int i = 0; i < height*width*channels; ++i) {
        if(count[i] > 0) {
            output[i] = input[i]/count[i];
        }else {
            output[i] = input[i];
        }
    }
    delete count;
}

#ifdef CPU_ONLY
STUB_GPU(NormalizedDeconvolutionLayer);
#endif

INSTANTIATE_CLASS(NormalizedDeconvolutionLayer);
REGISTER_LAYER_CLASS(NormalizedDeconvolution);

}  // namespace caffe
