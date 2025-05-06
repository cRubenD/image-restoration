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
            uchar grayPixel =  uchar((float) (pixel[0] + pixel[1] + pixel[2]) / 3.0f);
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

Mat adaptiveNoiseThreshold(const Mat& image, int windowSize, float k) {
    Mat result = Mat::zeros(image.size(), CV_8UC1);
    int halfWindow = windowSize / 2;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            float sum = 0;
            int count = 0;

            for (int wy = -halfWindow; wy <= halfWindow; wy++) {
                int ny = y + wy;
                if (ny < 0 || ny >= image.rows) continue;

                for (int wx = -halfWindow; wx <= halfWindow; wx++) {
                    int nx = x + wx;
                    if (nx < 0 || nx >= image.cols) continue;

                    sum += (float) image.at<uchar>(ny, nx);
                    count++;
                }
            }

            float mean = count > 0 ? sum / (float) count : 0;

            float variance = 0;
            for (int wy = -halfWindow; wy <= halfWindow; wy++) {
                int ny = y + wy;
                if (ny < 0 || ny >= image.rows) continue;

                for (int wx = -halfWindow; wx <= halfWindow; wx++) {
                    int nx = x + wx;
                    if (nx < 0 || nx >= image.cols) continue;

                    float diff = (float) image.at<uchar>(ny, nx) - mean;
                    variance += diff * diff;
                }
            }

            float stdDev = count > 0 ? sqrt(variance / (float) count) : 0;

            float threshold = mean + k * stdDev;
            result.at<uchar>(y, x) = ((float) image.at<uchar>(y, x) > threshold) ? 255 : 0;
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

Vec3b hsv2bgr(const Vec3b& hsv) {
    float h = (float) hsv[0] * 2.0f;  // H: [0, 180] -> [0, 360]
    float s = (float) hsv[1] / 255.0f;
    float v = (float) hsv[2] / 255.0f;

    float c = v * s;
    float x = (float) c * (1 - fabs(fmod(h / 60.0f, 2) - 1));
    float m = v - c;

    float r, g, b;
    if (h >= 0 && h < 60) {
        r = c; g = x; b = 0;
    } else if (h >= 60 && h < 120) {
        r = x; g = c; b = 0;
    } else if (h >= 120 && h < 180) {
        r = 0; g = c; b = x;
    } else if (h >= 180 && h < 240) {
        r = 0; g = x; b = c;
    } else if (h >= 240 && h < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }

    return {
            static_cast<uchar>((b + m) * 255),
            static_cast<uchar>((g + m) * 255),
            static_cast<uchar>((r + m) * 255)
    };
}

void morphErode(const Mat& src, Mat& dst, const Mat& kernel) {
    dst = Mat::zeros(src.size(), src.type());
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int kernelCenterX = kernelCols / 2;
    int kernelCenterY = kernelRows / 2;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            bool valid = true;

            for (int ky = 0; ky < kernelRows && valid; ky++) {
                for (int kx = 0; kx < kernelCols && valid; kx++) {
                    if (kernel.at<uchar>(ky, kx) > 0) {
                        int ny = y + (ky - kernelCenterY);
                        int nx = x + (kx - kernelCenterX);

                        if (ny >= 0 && ny < src.rows && nx >= 0 && nx < src.cols) {
                            if (src.at<uchar>(ny, nx) == 0) {
                                valid = false;
                            }
                        }
                    }
                }
            }

            dst.at<uchar>(y, x) = valid ? 255 : 0;
        }
    }
}

// Custom implementation of dilation
void morphDilate(const Mat& src, Mat& dst, const Mat& kernel) {
    dst = Mat::zeros(src.size(), src.type());
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int kernelCenterX = kernelCols / 2;
    int kernelCenterY = kernelRows / 2;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            bool hit = false;

            for (int ky = 0; ky < kernelRows && !hit; ky++) {
                for (int kx = 0; kx < kernelCols && !hit; kx++) {
                    if (kernel.at<uchar>(ky, kx) > 0) {
                        int ny = y + (ky - kernelCenterY);
                        int nx = x + (kx - kernelCenterX);

                        if (ny >= 0 && ny < src.rows && nx >= 0 && nx < src.cols) {
                            if (src.at<uchar>(ny, nx) > 0) {
                                hit = true;
                            }
                        }
                    }
                }
            }

            dst.at<uchar>(y, x) = hit ? 255 : 0;
        }
    }
}


