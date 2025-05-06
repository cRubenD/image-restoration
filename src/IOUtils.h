//
// Created by ruben on 4/29/2025.
//

#ifndef LAB6_IOUTILS_H
#define LAB6_IOUTILS_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat loadImage(const String& path);

void saveImage(const String& path, const Mat& image);

void displayImage(const String& windowName, const Mat& image);

void displayComparison(const String& windowName, const Mat& image1, const Mat& image2);

#endif //LAB6_IOUTILS_H
