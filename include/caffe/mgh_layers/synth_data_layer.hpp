#ifndef SYNTH_DATA_LAYER_HPP_VUYPEMGF
#define SYNTH_DATA_LAYER_HPP_VUYPEMGF


#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define RAND_GEN_M_PI 3.141592653589793238462643383279502884L

namespace caffe {

/** Generate synthetic patches.
 *
 * All OpenCV cv::Mat returned or expected by the following public functions are
 * of size (sz,sz) and type CV_32FC3 where pixel is Vec3f
 *
 \code
 * SyntheticPatchGenerator synth(seed);  // default seed = system time
 *
 * cv::Mat synth.get_sine_sample   (int32_t sz);
 * cv::Mat synth.get_stroke_sample (int32_t sz);
 * cv::Mat synth.get_ellipse_sample(int32_t sz);
 * cv::Mat synth.save              (std::string filename, cv::Mat img);
 *
 \endcode
 */
class SyntheticPatchGenerator {
    public:
        SyntheticPatchGenerator(int32_t seed=0);

        cv::Mat get_sine_sample(int32_t sz);
        cv::Mat get_stroke_sample(int32_t sz);
        cv::Mat get_ellipse_sample(int32_t sz);

        void save(std::string filename, cv::Mat patch);

    private:

        // -- Random numbers -----------------------------------------------------------
        int32_t randint(int32_t min_val=0, int max_val=0);
        double uniform_random(void);
        double normal_random(void);

        cv::Mat randn(int32_t rows, int32_t cols);
        cv::Mat rand_3(int32_t rows, int32_t cols);
        cv::Mat randn_3(int32_t rows, int32_t cols);

        // --- Utils -------------------------------------------------------------------

        template<class T> 
        const T& clamp(const T& x, const T& lower, const T& upper) {
            return std::min<T>(upper, std::max<T>(x, lower));
        }
        float round(float v);

        cv::Mat rgb2hsv(cv::Mat in);
        cv::Mat hsv2rgb(cv::Mat in);

        void assign(cv::Mat& m, int32_t rstart, int32_t rend, int32_t cstart, int32_t cend, float val);
        void assign(cv::Mat& m, int32_t rstart, int32_t rend, int32_t cstart, int32_t cend, cv::Mat val);

        cv::Mat rotate(cv::Mat source, float angle);
        void gaussian_blur(cv::Mat& m, float sigma);

        // -----------------------------------------------------------------------------

        cv::Mat get_stroke_mask(int32_t sz, int32_t width, int32_t angle);
        cv::Mat get_ellipse_mask(int32_t sz, int32_t r, int32_t c, int32_t radius_1, int32_t radius_2, int32_t boundary_only);
        cv::Mat get_sine_mask(int32_t sz, float angle);
        cv::Mat colorize_sample(cv::Mat mask, cv::Vec3f fg, cv::Vec3f bg);

        void add_noise(cv::Mat& sample);

        std::pair<cv::Vec3f,cv::Vec3f> get_color_sample(void);

        cv::Mat convert_to_byte(cv::Mat img_f);

};

template <typename Dtype>
class SynthDataLayer : public Layer<Dtype> {
    public:
        explicit SynthDataLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        // Data layers should be shared by multiple solvers in parallel
        virtual inline bool ShareInParallel() const { return true; }
        // Data layers have no bottoms, so reshaping is trivial.
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {}
        virtual inline const char* type() const { return "SynthData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        vector<string> types_;
        SyntheticPatchGenerator generator_;

        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { NOT_IMPLEMENTED; }
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top)  { NOT_IMPLEMENTED; }
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { NOT_IMPLEMENTED; }
};

}  // namespace caffe

#endif /* end of include guard: SYNTH_DATA_LAYER_HPP_VUYPEMGF */

