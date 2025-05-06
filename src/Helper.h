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

vector<vector<float>> makeGaussianKernel(int kernelSize, float sigma);

Vec3b bgr2hsv(const Vec3b& bgr);

#endif //LAB6_HELPER_H
