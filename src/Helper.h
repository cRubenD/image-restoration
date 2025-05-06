//
// Created by ruben on 5/5/2025.
//

#ifndef LAB6_HELPER_H
#define LAB6_HELPER_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat convertToGrayscale(const Mat& image);

Mat applyThreshold(const Mat& image, uchar threshold);

Mat adaptiveNoiseThreshold(const Mat& image, int windowSize, float k);

vector<vector<float>> makeGaussianKernel(int kernelSize, float sigma);

Vec3b bgr2hsv(const Vec3b& bgr);

Vec3b hsv2bgr(const Vec3b& hsv);

void morphErode(const Mat& src, Mat& dst, const Mat& kernel);

void morphDilate(const Mat& src, Mat& dst, const Mat& kernel);

#endif //LAB6_HELPER_H
