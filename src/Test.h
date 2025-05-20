//
// Created by ruben on 5/20/2025.
//

#ifndef LAB6_TEST_H
#define LAB6_TEST_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat doInpaintWithOpenCV(const Mat& src);

Mat doDenoiseWithOpenCV(const Mat& src);

Mat doColorCorrectionWithOpenCV(const Mat& src);

Mat detectScratchesOpenCV(const Mat& src);

Mat detectDamageRegionsOpenCV(const Mat& src);

Mat detectNoiseOpenCV(const Mat& src);

#endif //LAB6_TEST_H
