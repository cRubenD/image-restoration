//
// Created by ruben on 4/29/2025.
//

#ifndef LAB6_RESTORER_H
#define LAB6_RESTORER_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

using RestoreOp = function<Mat(const Mat&)>;

Mat customInpaint(const Mat& image, const Mat& mask, int radius);

Mat reduceNoise(const Mat& image, const Mat& noiseMask);

Mat restoreColorDegradation(const Mat& image, const Mat& colorDegradationMask);

Mat doInpaint(const Mat& src);

Mat doDenoise(const Mat& src);

Mat doColor(const Mat& src);


#endif //LAB6_RESTORER_H
