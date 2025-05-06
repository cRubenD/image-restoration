//
// Created by ruben on 4/29/2025.
//

#ifndef LAB6_RESTORER_H
#define LAB6_RESTORER_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat inpaintDamage(const Mat& image, const Mat& mask);

Mat removeScratchLines(const Mat& image, const Mat& lineMask);

Mat restoreDamagedRegions(const Mat& image, const Mat& damageMask);

Mat reduceNoise(const Mat& image, const Mat& noiseMask);

Mat restoreColorDegradation(const Mat& image, const Mat& colorDegradationMask);

#endif //LAB6_RESTORER_H
