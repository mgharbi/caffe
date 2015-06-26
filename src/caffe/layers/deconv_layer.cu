#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const Dtype* weight = this->blobs_[0]->gpu_data();
    bool doNormalize = true;
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->backward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                    top_data + top[i]->offset(n));
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
            }
            if(doNormalize){ // Normalize boundaries
                this->normalize_boundaries_gpu(top_data + top[i]->offset(n), top_data + top[i]->offset(n));
            }
        }
    }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->gpu_diff();
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

        bool doNormalize = true;

        // Bias gradient, if necessary.
        if (this->bias_term_ && this->param_propagate_down_[1]) {
            Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
            vector<int> shape(4);
            shape[0] = 1;
            shape[1] = top[i]->shape(1);
            shape[2] = top[i]->shape(2);
            shape[3] = top[i]->shape(3);
            Blob<Dtype> normalized(shape);
            Dtype * norm_diff = normalized.mutable_gpu_data();
            for (int n = 0; n < this->num_; ++n) {
                if(doNormalize) {
                    this->normalize_boundaries_gpu(top_diff + top[i]->offset(n), norm_diff);
                    this->backward_gpu_bias(bias_diff, norm_diff);
                } else {
                    this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
                }
            }
        }
        if (this->param_propagate_down_[0] || propagate_down[i]) {
            vector<int> shape(4);
            shape[0] = 1;
            shape[1] = top[i]->shape(1);
            shape[2] = top[i]->shape(2);
            shape[3] = top[i]->shape(3);
            Blob<Dtype> normalized(shape);
            Dtype * norm_diff = normalized.mutable_gpu_data();
            for (int n = 0; n < this->num_; ++n) {
                if(doNormalize) {
                    this->normalize_boundaries_gpu(top_diff + top[i]->offset(n), norm_diff);
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0]) {
                        this->weight_gpu_gemm(norm_diff,
                                bottom_data + bottom[i]->offset(n), weight_diff);
                    }
                    // gradient w.r.t. bottom data, if necessary.
                    if (propagate_down[i]) {
                        this->forward_gpu_gemm(norm_diff, weight,
                                bottom_diff + bottom[i]->offset(n));
                    }
                } else {
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0]) {
                        this->weight_gpu_gemm(top_diff + top[i]->offset(n),
                                bottom_data + bottom[i]->offset(n), weight_diff);
                    }
                    // gradient w.r.t. bottom data, if necessary.
                    if (propagate_down[i]) {
                        this->forward_gpu_gemm(top_diff + top[i]->offset(n), weight,
                                bottom_diff + bottom[i]->offset(n));
                    }
                }
            }
        }
    }
}

template <typename Dtype>
__global__ void normalize_boundaries_gpu_kernel(const Dtype* input, Dtype* output,const int n,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col
)
{
  CUDA_KERNEL_LOOP(index, n) {
    // Dtype val = 0;
    int vcount = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    // int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);

    // int offset = (c * kernel_h * kernel_w + h * kernel_w + w) * height_col * width_col;
    // int coeff_h_col = (1 - stride_h * kernel_w * height_col) * width_col;
    // int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
        vcount += 1;
      }
    }
    // data_im[index] = val;
    if(vcount > 0) {
        output[index] = input[index]/ vcount;
    } else {
        output[index] = input[index];
    }
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::normalize_boundaries_gpu(const Dtype* input, Dtype* output)
{
    int channels = this->conv_in_channels_;
    int height   = this->conv_in_height_;
    int width    = this->conv_in_width_;
    int pad_h    = this->pad_h_;
    int pad_w    = this->pad_w_;
    int stride_h = this->stride_h_;
    int stride_w = this->stride_w_;
    int kernel_h = this->kernel_h_;
    int kernel_w = this->kernel_w_;

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int num_kernels = channels * height * width;
    normalize_boundaries_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
        CAFFE_CUDA_NUM_THREADS>>>(input, output,
                num_kernels, height, width, kernel_h, kernel_w, pad_h,
                pad_w, stride_h, stride_w, height_col, width_col);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(DeconvolutionLayer);

}  // namespace caffe
