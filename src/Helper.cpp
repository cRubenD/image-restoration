//
// Created by ruben on 5/5/2025.
//

#include "Helper.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat convertToGrayscale(const Mat& image) {

    Mat result = Mat(image.rows, image.cols, CV_8UC1);

    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            uchar grayPixel = (pixel[0] + pixel[1] + pixel[2]) / 3;
            result.at<uchar>(y, x) = grayPixel;
        }
    }

    return result;
}

Mat applyThreshold(const Mat& image, uchar threshold) {

    Mat result(image.rows, image.cols, CV_8UC1);
    for(int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            result.at<uchar>(y, x) = (image.at<uchar>(y, x) > threshold) ? 255 : 0;
        }
    }

    return result;
}

vector<vector<float>> makeGaussianKernel(int kernelSize, float sigma) {
    vector<vector<float>> kernel(kernelSize, vector<float>(kernelSize));
    float sum = 0.0f;
    int halfSize = kernelSize / 2;

    for(int y = -halfSize; y <= halfSize; y++) {
        for(int x = -halfSize; x <= halfSize; x++) {
            float value = exp((float) -(x * x + y * y) / (2 * sigma * sigma));
            kernel[y + halfSize][x + halfSize] = value;
            sum += value;
        }
    }

    for (auto &row : kernel) {
        for (float &val : row) {
            val /= sum;
        }
    }

    return kernel;
}

Vec3b bgr2hsv(const Vec3b& bgr) {
    float b = (float) bgr[0] / 255.0f;
    float g = (float) bgr[1] / 255.0f;
    float r = (float) bgr[2] / 255.0f;
    float mx = max(r, max(g, b));
    float mn = min(r, min(g, b));
    float diff = mx - mn;

    float h = 0, s = 0, v = mx;
    if (diff > 1e-6) {
        s = diff / mx;
        if (mx == r)      h = fmodf((g - b) / diff, 6.0f);
        else if (mx == g) h = (b - r) / diff + 2.0f;
        else               h = (r - g) / diff + 4.0f;
        h *= 60.0f;
        if (h < 0) h += 360.0f;
    }

    return {
            static_cast<uchar>(h / 2.0f),         // H: [0,180]
            static_cast<uchar>(s * 255.0f),       // S: [0,255]
            static_cast<uchar>(v * 255.0f)        // V: [0,255]
    };
}