#include <vector>

#include "caffe/mgh_layers/zipper_metric_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ZipperMetricLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 3)
      << "Bottom blobs must have 3 channels";
  CHECK_EQ(bottom[1]->channels(), 3)
      << "Bottom blobs must have 3 channels";
  threshold_ = 2.3; // JND for Lab
}

template <typename Dtype>
void ZipperMetricLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void ZipperMetricLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* source = bottom[0]->cpu_data();
    const Dtype* ref = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int n = bottom[0]->num();
    int c = bottom[0]->channels();
    int h = bottom[0]->height();
    int w = bottom[0]->width();
    float zratio = 0.0f;
    float mean_zratios = 0.0f;
    for (int i = 0; i < n; ++i) {
        int offset = bottom[0]->offset(i,0,0,0);
        zipper_ratio(c, h, w, source+offset, ref+offset, zratio);
        mean_zratios += zratio;
    }
    mean_zratios /= n;
    top_data[0] = mean_zratios;

}

template <typename Dtype>
void ZipperMetricLayer<Dtype>::zipper_ratio(
    int chan, int height, int width,
    const Dtype* src,
    const Dtype* ref,
    float &zratio
){
    int nPix = height*width;

    int nZippered = 0;
    // Get the most similar neighbor in the reference image
    for (int y = 1; y < height-1; y++)
    for (int x = 1; x < width-1; x++)
    {
        Dtype min_delta = FLT_MAX;
        int min_x = x;
        int min_y = y;

        // Visit neighbors
        for (int y2 = y-1; y2 <= y+1; y2++)
        for (int x2 = x-1; x2 <= x+1; x2++)
        {
            if(x2 == x && y2 == y) continue; // skip self

            Dtype delta = Dtype(0); // difference value
            for (int z = 0; z < chan; z++) {
                Dtype val = ref[x + width*y + width*height*z]
                          - ref[x2 + width*y2 + width*height*z];
                delta += val*val;
            }
            if( delta < min_delta ) {
                min_delta = delta;
                min_x = x2;
                min_y = y2;
            }
        } // Now the most similar neighbor in the reference is min_x,min_y

        // Get the difference in the new image at this location
        Dtype src_delta = Dtype(0);
        for (int z = 0; z < chan; z++) {
            Dtype val = src[x + width*y + width*height*z]
                      - src[min_x + width*min_y + width*height*z];
            src_delta += val*val;
        }

        if(sqrt(src_delta)-sqrt(min_delta) > threshold_){
            // There's a contrast increase in the closest neighbor higher than
            // the threshold: count this pixel as zippered
            nZippered++;
        }
    }

    zratio = (float) nZippered / (float) nPix;
}

#ifdef CPU_ONLY
STUB_GPU(ZipperMetricLayer);
#endif

INSTANTIATE_CLASS(ZipperMetricLayer);
REGISTER_LAYER_CLASS(ZipperMetric);

}  // namespace caffe
