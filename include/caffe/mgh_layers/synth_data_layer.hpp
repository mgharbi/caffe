#ifndef SYNTH_DATA_LAYER_HPP_VUYPEMGF
#define SYNTH_DATA_LAYER_HPP_VUYPEMGF


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

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
class SyntheticPatchGenerator;

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
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class caffe::SyntheticPatchGenerator {

#define RAND_GEN_M_PI 3.141592653589793238462643383279502884L

    private:

        template<class T> const T& clamp(const T& x, const T& lower, const T& upper) {
            return std::min<T>(upper, std::max<T>(x, lower));
        }

        int32_t randint(int32_t min_val=0, int max_val=0) {
            int32_t r = rand();
            if (max_val>0) {
                r = min_val + r%max_val;
            } else {
                if (min_val>0) {
                    r = r%min_val;
                }
            }
            return r;
        }

        double uniform_random(void) {
            return double(rand()) / double(RAND_MAX-1);
        }

        double normal_random(void) {
            double pi = RAND_GEN_M_PI;
            double u1 = uniform_random();
            double u2 = uniform_random();
            double z0 = std::sqrt(-2.0*std::log(u1)) * std::cos(2*pi*u2);
            double z1 = std::sqrt(-2.0*std::log(u1)) * std::sin(2*pi*u2);
            return (randint(2)==0 ? z0 : z1);
        }

        cv::Mat randn(int32_t rows, int32_t cols) {
            cv::Mat m = cv::Mat(rows, cols, CV_32F);
            for (int32_t i=0; i<rows; i++) {
                for (int32_t j=0; j<cols; j++) {
                    m.at<float>(i,j) = normal_random();
                }
            }
            return m;
        }

        cv::Mat rand_3(int32_t rows, int32_t cols) {
            cv::Mat m = cv::Mat(rows, cols, CV_32FC3);
            for (int32_t i=0; i<rows; i++) {
                for (int32_t j=0; j<cols; j++) {
                    m.at<cv::Vec3f>(i,j)[0] = uniform_random();
                    m.at<cv::Vec3f>(i,j)[1] = uniform_random();
                    m.at<cv::Vec3f>(i,j)[2] = uniform_random();
                }
            }
            return m;
        }

        cv::Mat randn_3(int32_t rows, int32_t cols) {
            cv::Mat m = cv::Mat(rows, cols, CV_32FC3);
            for (int32_t i=0; i<rows; i++) {
                for (int32_t j=0; j<cols; j++) {
                    m.at<cv::Vec3f>(i,j)[0] = normal_random();
                    m.at<cv::Vec3f>(i,j)[1] = normal_random();
                    m.at<cv::Vec3f>(i,j)[2] = normal_random();
                }
            }
            return m;
        }

        float round(float v) {
            return std::floor(v+0.5f);
        }

        // -----------------------------------------------------------------------------

        cv::Mat rgb2hsv(cv::Mat in) {
            cv::Mat out;
            cvtColor(in, out, CV_RGB2HSV);
            return out;
        }

        cv::Mat hsv2rgb(cv::Mat in) {
            cv::Mat out;
            cvtColor(in, out, CV_HSV2RGB);
            return out;
        }

        void assign(cv::Mat& m, int32_t rstart, int32_t rend, int32_t cstart, int32_t cend, float val) {
            int32_t rend_t = (rend>0 ? rend : m.rows-rend);
            int32_t cend_t = (cend>0 ? cend : m.cols-cend);
            for (int32_t i=rstart; i<rend_t; i++) {
                for (int32_t j=cstart; j<cend_t; j++) {
                    m.at<float>(i,j) = val;
                }
            }
        }

        void assign(cv::Mat& m, int32_t rstart, int32_t rend, int32_t cstart, int32_t cend, cv::Mat val) {
            int32_t rend_t = (rend>0 ? rend : m.rows-rend);
            int32_t cend_t = (cend>0 ? cend : m.cols-cend);
            for (int32_t i=rstart; i<rend_t; i++) {
                for (int32_t j=cstart; j<cend_t; j++) {
                    m.at<float>(i,j) = val.at<float>(i-rstart, j-cstart);
                }
            }
        }

        cv::Mat rotate(cv::Mat source, float angle) {
            cv::Point2f src_center(source.cols/2.0f, source.rows/2.0f);
            cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
            cv::Mat dst;
            warpAffine(source, dst, rot_mat, source.size());
            return dst;
        }

        void gaussian_blur(cv::Mat& m, float sigma) {
            cv::Mat m_filtered = cv::Mat::zeros(m.rows, m.cols, m.type());
            cv::GaussianBlur(m, m_filtered, cv::Size(0,0), sigma);
            m = m_filtered;
        }

        // -----------------------------------------------------------------------------

        cv::Mat get_stroke_mask(int32_t sz, int32_t width, int32_t angle) {
            cv::Mat stroke = cv::Mat::ones(sz, sz, CV_32F);
            assign(stroke, sz/4,-sz/4, sz/2-width/2,sz/2-width/2+width, 0);
            cv::Mat rotated = rotate(stroke, angle);
            return rotated;
        }

        cv::Mat get_ellipse_mask(int32_t sz, int32_t r, int32_t c, int32_t radius_1, int32_t radius_2, int32_t boundary_only) {
            cv::Mat sample = cv::Mat::ones(sz, sz, CV_32F);
            ellipse(sample,
                    cv::Point(r,c),          // center
                    cv::Size(radius_1, radius_2),// axes
                    0,                      // angle,
                    0,                      // startAngle,
                    360,                    // endAngle,
                    0,                      // color
                    !boundary_only ? 1: -1  // thickness: should be positive for boundary only
                   );
            return sample;
        }

        cv::Mat get_sine_mask(int32_t sz, float angle) {
            float pi = RAND_GEN_M_PI;

            int32_t h = sz;
            int32_t w = sz;

            float start_pulse = 2*pi/64.0;
            float end_pulse   = 2*pi/4.0;
            float b = start_pulse;
            float a = (end_pulse-b)/(w*std::sqrt(2));

            cv::Mat im = cv::Mat::zeros(h, w, CV_32F);
            for (int32_t i=0; i<im.rows; i++) {
                for (int32_t j=0; j<im.cols; j++) {
                    float x = j;
                    float y = i;
                    float var = std::cos(2*pi*angle/360.0)*x + std::sin(2*pi*angle/360.0)*y;
                    im.at<float>(i,j) = 0.5+std::sin(var*(a*var+b))/2;
                }
            }

            return im;
        }

        cv::Mat colorize_sample(cv::Mat mask, cv::Vec3f fg, cv::Vec3f bg) {
            cv::Mat color = cv::Mat::zeros(mask.rows, mask.cols, CV_32FC3);
            for (int32_t i=0; i<mask.rows; i++) {
                for (int32_t j=0; j<mask.cols; j++) {
                    float m = mask.at<float>(i,j);
                    cv::Vec3f v = m*bg + (1.0-m)*fg;
                    color.at<cv::Vec3f>(i,j) = v;
                }
            }
            return color;
        }

        void add_noise(cv::Mat& sample) {
            int32_t monochromatic = randint(2);     // 0 or 1
            float   sigma   = uniform_random()*0.1; // uniform distributed in [0,0.1]
            int32_t sigma_s = randint(9);           // uniform distributed in [0,8]
            int32_t rows    = sample.rows;
            int32_t cols    = sample.cols;

            // gaussian noise mu=0, sigma=1
            cv::Mat noise = (monochromatic ? randn(rows, cols) : randn_3(rows, cols));

            if (sigma_s > 0) {
                gaussian_blur(noise, sigma_s);
            }

            for (int32_t i=0; i<rows; i++) {
                for (int32_t j=0; j<cols; j++) {
                    if (monochromatic) {
                        sample.at<cv::Vec3f>(i,j)[0] += sigma * noise.at<float>(i,j);
                        sample.at<cv::Vec3f>(i,j)[1] += sigma * noise.at<float>(i,j);
                        sample.at<cv::Vec3f>(i,j)[2] += sigma * noise.at<float>(i,j);
                    } else {
                        sample.at<cv::Vec3f>(i,j)[0] += sigma * noise.at<cv::Vec3f>(i,j)[0];
                        sample.at<cv::Vec3f>(i,j)[1] += sigma * noise.at<cv::Vec3f>(i,j)[1];
                        sample.at<cv::Vec3f>(i,j)[2] += sigma * noise.at<cv::Vec3f>(i,j)[2];
                    }
                }
            }

            for (int32_t i=0; i<rows; i++) {
                for (int32_t j=0; j<cols; j++) {
                    sample.at<cv::Vec3f>(i,j)[0] = clamp(sample.at<cv::Vec3f>(i,j)[0], 0.0f, 1.0f);
                    sample.at<cv::Vec3f>(i,j)[1] = clamp(sample.at<cv::Vec3f>(i,j)[1], 0.0f, 1.0f);
                    sample.at<cv::Vec3f>(i,j)[2] = clamp(sample.at<cv::Vec3f>(i,j)[2], 0.0f, 1.0f);
                }
            }
        }

        std::pair<cv::Vec3f,cv::Vec3f> get_color_sample(void) {
            cv::Mat fg = rand_3(1,1);
            cv::Mat bg;

            if (randint(2) == 0) {
                // monochrome
                bg = fg;
                if (randint(2) == 0) {
                    bg = rgb2hsv(bg);
                    bg.at<cv::Vec3f>(0,0)[1] *= uniform_random(); // saturation
                    bg.at<cv::Vec3f>(0,0)[2] *= uniform_random(); // brightness
                    bg = hsv2rgb(bg);
                }
                else {
                    fg = rgb2hsv(fg);
                    fg.at<cv::Vec3f>(0,0)[1] *= uniform_random(); // saturation
                    fg.at<cv::Vec3f>(0,0)[2] *= uniform_random(); // brightness
                    fg = hsv2rgb(fg);
                }
            } else {
                bg = rand_3(1,1);
            }

            return std::make_pair(fg.at<cv::Vec3f>(0,0), bg.at<cv::Vec3f>(0,0));
        }

        cv::Mat convert_to_byte(cv::Mat img_f) {
            cv::Mat img_u;
            img_f.convertTo(img_u, CV_8UC3, 255.0);
            return img_u;
        }

    public:

        SyntheticPatchGenerator(int32_t seed=0) {
            int32_t random_seed = seed;
            if (random_seed==0) {
                random_seed = time(NULL);
            }
            srand(seed);
        }

        cv::Mat get_sine_sample(int32_t sz) {
            float angle = uniform_random()*180;
            if (randint(2) == 0) { // quantize
                angle /= 45;
                angle = round(angle) * 45;
            }

            cv::Mat mask = get_sine_mask(sz,angle);
            std::pair<cv::Vec3f,cv::Vec3f> fb = get_color_sample();

            cv::Vec3f fg = fb.first;
            cv::Vec3f bg = fb.second;

            cv::Mat color = colorize_sample(mask, fg, bg);
            add_noise(color);

            return color;
        }

        cv::Mat get_stroke_sample(int32_t sz) {
            float angle = uniform_random()*180;
            if (randint(2) == 0) { // quantize
                angle /= 45;
                angle = round(angle) * 45;
            }

            int32_t width = randint(1,8);

            cv::Mat mask = get_stroke_mask(sz, width, angle);
            std::pair<cv::Vec3f,cv::Vec3f> fb = get_color_sample();

            cv::Vec3f fg = fb.first;
            cv::Vec3f bg = fb.second;

            cv::Mat color = colorize_sample(mask, fg, bg);
            add_noise(color);

            return color;
        }

        cv::Mat get_ellipse_sample(int32_t sz) {
            int32_t r = randint(2)+sz/2;
            int32_t c = randint(2)+sz/2;
            int32_t boundary_only = randint(2);
            int32_t radius1 = randint(sz/2);
            int32_t radius2 = randint(sz/2);
            int32_t sigma_s = randint(9);

            cv::Mat mask = get_ellipse_mask(sz, r, c, radius1, radius2, boundary_only);
            if (sigma_s > 0) {
                gaussian_blur(mask, sigma_s);
            }
            std::pair<cv::Vec3f,cv::Vec3f> fb = get_color_sample();

            cv::Vec3f fg = fb.first;
            cv::Vec3f bg = fb.second;

            cv::Mat color = colorize_sample(mask, fg, bg);
            add_noise(color);

            return color;
        }

        void save(std::string filename, cv::Mat patch) {
            cv::imwrite(filename, convert_to_byte(patch));
        }

#undef RAND_GEN_M_PI
};

#endif /* end of include guard: SYNTH_DATA_LAYER_HPP_VUYPEMGF */

