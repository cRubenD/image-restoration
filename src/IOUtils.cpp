//
// Created by ruben on 4/29/2025.
//

#include "IOUtils.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat loadImage(const String& path) {
    Mat image = imread(path, IMREAD_COLOR);
    if(image.empty()) throw runtime_error("Imaginea nu a putut fi incarcata!");

    return image;
}

void saveImage(const String& path, const Mat& image) {
    imwrite(path, image);
}

void displayImage(const String& windowName, const Mat& image) {
    namedWindow(windowName, WINDOW_GUI_NORMAL);
    imshow(windowName, image);
}

void displayComparison(const String& windowName, const Mat& image1, const Mat& image2) {

    int height = max(image1.rows, image2.rows);
    Mat comparison(height, image1.cols + image2.cols, image1.type());

    Mat leftImage = comparison(Rect(0, 0, image1.cols, image1.rows));
    image1.copyTo(leftImage);

    Mat rightImage = comparison(Rect(image1.cols, 0, image2.cols, image2.rows));
    image2.copyTo(rightImage);

    displayImage(windowName, comparison);
}