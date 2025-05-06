//
// Created by ruben on 4/29/2025.
//

#ifndef LAB6_DETECTOR_H
#define LAB6_DETECTOR_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

extern float sobelX[3][3];

extern float sobelY[3][3];

Mat detectScratchLines(const Mat& image);

Mat detectDamageRegions(const Mat& image);

Mat detectNoise(const Mat& image);

Mat detectColorDegradation(const Mat& image);

Mat detectAllDegradations();

void displayDegradationMasks();

Mat applySobelFilter(const Mat& grayscaleImage);

pair<Scalar, Scalar> calculateLocalStatistics(const Mat& image, int x, int y, int windowSize);

#endif //LAB6_DETECTOR_H
