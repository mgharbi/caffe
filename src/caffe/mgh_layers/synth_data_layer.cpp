#include <vector>

#include "caffe/filler.hpp"
#include "caffe/mgh_layers/synth_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SynthDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) 
{

    const SynthDataParameter& param = this->layer_param_.synth_data_param();

    CHECK(param.shape().dim(1) == 3)
        << " SynthDataLayer generates 3 channel images";
    CHECK(param.shape().dim(2) == param.shape().dim(3))
        << " SynthDataLayer generates square patches";

    top[0]->Reshape(param.shape());

    const int num_types = param.type_size();
    CHECK(num_types > 0) << "SynthDataLayer should have at least one type"
        << " SynthDataLayer generates square patches";
    for (int i = 0; i < num_types; ++i)
    {
        types_.push_back(param.type(i));
    }

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
    int h = shape[2];
    cv::Mat sample;
    for (int n = 0; n < shape[0]; ++n) { // each sample 

        unsigned int type_id = caffe_rng_rand() % types_.size();

        if (types_[type_id] == "sine") {
            sample = generator_.get_sine_sample(h);
        } else if (types_[type_id] == "stroke") {
            sample = generator_.get_stroke_sample(h);
        } else if (types_[type_id] == "ellipse") {
            sample = generator_.get_ellipse_sample(h);
        } else {
            NOT_IMPLEMENTED;
        }


        for (int32_t i=0; i<h; i++) {
            for (int32_t j=0; j<h; j++) {
                cv::Vec3f rgb = sample.at<cv::Vec3f>(i,j);
                top_data[top[0]->offset(n,0,j,i)] = rgb(0);
                top_data[top[0]->offset(n,1,j,i)] = rgb(1);
                top_data[top[0]->offset(n,2,j,i)] = rgb(2);
            }
        }
    }
}

INSTANTIATE_CLASS(SynthDataLayer);
REGISTER_LAYER_CLASS(SynthData);


// --------------------- Generator -------------------------------------------


SyntheticPatchGenerator::SyntheticPatchGenerator(int32_t seed)
{
    int32_t random_seed = seed;
    if (random_seed==0) {
        random_seed = time(NULL);
    }
    srand(random_seed);
}

cv::Mat SyntheticPatchGenerator::get_sine_sample(int32_t sz) {
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

cv::Mat SyntheticPatchGenerator::get_stroke_sample(int32_t sz) {
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

cv::Mat SyntheticPatchGenerator::get_ellipse_sample(int32_t sz) {
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

void SyntheticPatchGenerator::save(std::string filename, cv::Mat patch) {
    cv::imwrite(filename, convert_to_byte(patch));
}


// -- Random numbers -----------------------------------------------------------
int32_t SyntheticPatchGenerator::randint(int32_t min_val, int max_val) {
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

double SyntheticPatchGenerator::uniform_random(void) {
    return double(rand()) / double(RAND_MAX-1);
}

double SyntheticPatchGenerator::normal_random(void) {
    double pi = RAND_GEN_M_PI;
    double u1 = uniform_random();
    double u2 = uniform_random();
    double z0 = std::sqrt(-2.0*std::log(u1)) * std::cos(2*pi*u2);
    double z1 = std::sqrt(-2.0*std::log(u1)) * std::sin(2*pi*u2);
    return (randint(2)==0 ? z0 : z1);
}

cv::Mat SyntheticPatchGenerator::randn(int32_t rows, int32_t cols) {
    cv::Mat m = cv::Mat(rows, cols, CV_32F);
    for (int32_t i=0; i<rows; i++) {
        for (int32_t j=0; j<cols; j++) {
            m.at<float>(i,j) = normal_random();
        }
    }
    return m;
}

cv::Mat SyntheticPatchGenerator::rand_3(int32_t rows, int32_t cols) {
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

cv::Mat SyntheticPatchGenerator::randn_3(int32_t rows, int32_t cols) {
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

// -- Utils --------------------------------------------------------------------

float SyntheticPatchGenerator::round(float v) {
    return std::floor(v+0.5f);
}

cv::Mat SyntheticPatchGenerator::rgb2hsv(cv::Mat in) {
    cv::Mat out;
    cvtColor(in, out, CV_RGB2HSV);
    return out;
}

cv::Mat SyntheticPatchGenerator::hsv2rgb(cv::Mat in) {
    cv::Mat out;
    cvtColor(in, out, CV_HSV2RGB);
    return out;
}

void SyntheticPatchGenerator::assign(cv::Mat& m, int32_t rstart, int32_t rend, int32_t cstart, int32_t cend, float val) {
    int32_t rend_t = (rend>0 ? rend : m.rows-rend);
    int32_t cend_t = (cend>0 ? cend : m.cols-cend);
    for (int32_t i=rstart; i<rend_t; i++) {
        for (int32_t j=cstart; j<cend_t; j++) {
            m.at<float>(i,j) = val;
        }
    }
}

void SyntheticPatchGenerator::assign(cv::Mat& m, int32_t rstart, int32_t rend, int32_t cstart, int32_t cend, cv::Mat val) {
    int32_t rend_t = (rend>0 ? rend : m.rows-rend);
    int32_t cend_t = (cend>0 ? cend : m.cols-cend);
    for (int32_t i=rstart; i<rend_t; i++) {
        for (int32_t j=cstart; j<cend_t; j++) {
            m.at<float>(i,j) = val.at<float>(i-rstart, j-cstart);
        }
    }
}

cv::Mat SyntheticPatchGenerator::rotate(cv::Mat source, float angle) {
    cv::Point2f src_center(source.cols/2.0f, source.rows/2.0f);
    cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    warpAffine(source, dst, rot_mat, source.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return dst;
}

void SyntheticPatchGenerator::gaussian_blur(cv::Mat& m, float sigma) {
    cv::Mat m_filtered = cv::Mat::zeros(m.rows, m.cols, m.type());
    cv::GaussianBlur(m, m_filtered, cv::Size(0,0), sigma);
    m = m_filtered;
}

// -----------------------------------------------------------------------------

cv::Mat SyntheticPatchGenerator::get_stroke_mask(int32_t sz, int32_t width, int32_t angle) {
    cv::Mat stroke = cv::Mat::ones(sz, sz, CV_32F);
    assign(stroke, sz/4,-sz/4, sz/2-width/2,sz/2-width/2+width, 0);
    cv::Mat rotated = rotate(stroke, angle);
    return rotated;
}

cv::Mat SyntheticPatchGenerator::get_ellipse_mask(int32_t sz, int32_t r, int32_t c, int32_t radius_1, int32_t radius_2, int32_t boundary_only) {
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

cv::Mat SyntheticPatchGenerator::get_sine_mask(int32_t sz, float angle) {
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

cv::Mat SyntheticPatchGenerator::colorize_sample(cv::Mat mask, cv::Vec3f fg, cv::Vec3f bg) {
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

void SyntheticPatchGenerator::add_noise(cv::Mat& sample) {
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

std::pair<cv::Vec3f,cv::Vec3f> SyntheticPatchGenerator::get_color_sample(void) {
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

cv::Mat SyntheticPatchGenerator::convert_to_byte(cv::Mat img_f) {
    cv::Mat img_u;
    img_f.convertTo(img_u, CV_8UC3, 255.0);
    return img_u;
}

}  // namespace caffe
