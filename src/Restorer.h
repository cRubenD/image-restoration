//
// Created by ruben on 4/29/2025.
//

#ifndef LAB6_RESTORER_H
#define LAB6_RESTORER_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat customInpaint(const Mat& image, const Mat& mask, int radius);

Mat removeScratchLines(const Mat& image, const Mat& lineMask);

Mat restoreDamagedRegions(const Mat& image, const Mat& damageMask);

Mat reduceNoise(const Mat& image, const Mat& noiseMask);

Mat restoreColorDegradation(const Mat& image, const Mat& colorDegradationMask);

Mat restoreImage(const Mat& image);


#endif //LAB6_RESTORER_H
