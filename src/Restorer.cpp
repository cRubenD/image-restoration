//
// Created by ruben on 4/29/2025.
//

#include "Restorer.h"
#include "Detector.h"
#include "Helper.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// inpainting bazat pe difuzie
Mat customInpaint(const Mat& image, const Mat& mask, int radius) {
    Mat result = image.clone();
    Mat workMask = mask.clone();

    // calculam frontiera prima data dintre pixelii deteriorati si cei buni
    vector<pair<int, int>> boundary;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (workMask.at<uchar>(y, x) > 0) {

                bool isBoundary = false;
                for (int dy = -1; dy <= 1 && !isBoundary; dy++) {
                    for (int dx = -1; dx <= 1 && !isBoundary; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < image.rows && nx >= 0 && nx < image.cols) {
                            if (workMask.at<uchar>(ny, nx) == 0) {
                                isBoundary = true;
                            }
                        }
                    }
                }
                if (isBoundary) {
                    boundary.push_back({y, x});
                }
            }
        }
    }

    while (!boundary.empty()) {
        vector<pair<int, int>> newBoundary;

        for (const auto& pixel : boundary) {
            int y = pixel.first;
            int x = pixel.second;

            if (workMask.at<uchar>(y, x) == 0) {
                continue;
            }

            // calculam media ponderata din vecinii buni
            Vec3f sum(0, 0, 0);
            float weightSum = 0;

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny >= 0 && ny < image.rows && nx >= 0 && nx < image.cols) {
                        if (workMask.at<uchar>(ny, nx) == 0) {
                            float weight = (float) 1.0f / (1.0f + sqrt(dx*dx + dy*dy));
                            Vec3b neighbor = result.at<Vec3b>(ny, nx);
                            sum[0] += weight * (float) neighbor[0];
                            sum[1] += weight * (float) neighbor[1];
                            sum[2] += weight * (float) neighbor[2];
                            weightSum += weight;
                        }
                    }
                }
            }

            if (weightSum > 0) {
                Vec3b newValue;
                newValue[0] = saturate_cast<uchar>(sum[0] / weightSum);
                newValue[1] = saturate_cast<uchar>(sum[1] / weightSum);
                newValue[2] = saturate_cast<uchar>(sum[2] / weightSum);
                result.at<Vec3b>(y, x) = newValue;
                workMask.at<uchar>(y, x) = 0;

                // am schimbat pixelul si acum ne extindem la venicii din frontiera
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < image.rows && nx >= 0 && nx < image.cols) {
                            if (workMask.at<uchar>(ny, nx) > 0) {
                                bool isNewBoundary = false;
                                for (int bdy = -1; bdy <= 1 && !isNewBoundary; bdy++) {
                                    for (int bdx = -1; bdx <= 1 && !isNewBoundary; bdx++) {
                                        int bny = ny + bdy;
                                        int bnx = nx + bdx;
                                        if (bny >= 0 && bny < image.rows && bnx >= 0 && bnx < image.cols) {
                                            if (workMask.at<uchar>(bny, bnx) == 0) {
                                                isNewBoundary = true;
                                            }
                                        }
                                    }
                                }
                                if (isNewBoundary) {
                                    newBoundary.push_back({ny, nx});
                                }
                            }
                        }
                    }
                }
            } else {
                newBoundary.push_back({y, x});
            }
        }

        boundary = newBoundary;
    }

    return result;
}

// filtru bilateral manual pentru netezirea imaginii in functie de greutatea spatiala si de culoare
// pentru reducerea zgomotului
Mat customBilateralFilter(const Mat& src, float sigmaSpace, float sigmaColor) {
    Mat dst = src.clone();
    int radius = static_cast<int>(sigmaSpace * 3.0f);

    // calculul greutatilor spatiale
    vector<float> spatialWeights((2 * radius + 1) * (2 * radius + 1));
    int idx = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist = (float) sqrt(dx * dx + dy * dy);
            spatialWeights[idx++] = exp(-(dist * dist) / (2 * sigmaSpace * sigmaSpace));
        }
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f sum(0, 0, 0);
            float totalWeight = 0;
            Vec3b centerPixel = src.at<Vec3b>(y, x);
            idx = 0;

            for (int dy = -radius; dy <= radius; dy++) {
                int ny = y + dy;
                if (ny < 0 || ny >= src.rows) {
                    idx += 2*radius + 1;
                    continue;
                }

                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= src.cols) {
                        idx++;
                        continue;
                    }

                    Vec3b currentPixel = src.at<Vec3b>(ny, nx);

                    float spaceWeight = spatialWeights[idx++];

                    // calculul distantei de culoare
                    float colorDist = 0;
                    for (int c = 0; c < 3; c++) {
                        float diff = (float) centerPixel[c] - (float) currentPixel[c];
                        colorDist += diff * diff;
                    }
                    colorDist = sqrt(colorDist);
                    float colorWeight = exp(-(colorDist*colorDist) / (2 * sigmaColor * sigmaColor));

                    float weight = spaceWeight * colorWeight;

                    for (int c = 0; c < 3; c++) {
                        sum[c] += weight * (float) currentPixel[c];
                    }
                    totalWeight += weight;
                }
            }

            if (totalWeight > 0) {
                for (int c = 0; c < 3; c++) {
                    dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(sum[c] / totalWeight);
                }
            }
        }
    }

    return dst;
}

Mat restoreColorDegradation(const Mat& image, const Mat& colorDegradationMask) {
    Mat result = image.clone();

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (colorDegradationMask.at<uchar>(y, x) > 0) {
                // se foloseste o fereastra de 7 x 7 si se calculeaza o medie a culorilor
                vector<Vec3b> samples;
                int windowSize = 7;
                int halfSize = windowSize / 2;

                for (int dy = -halfSize; dy <= halfSize; dy++) {
                    for (int dx = -halfSize; dx <= halfSize; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
                            if (colorDegradationMask.at<uchar>(ny, nx) == 0) {
                                samples.push_back(image.at<Vec3b>(ny, nx));
                            }
                        }
                    }
                }

                if (!samples.empty()) {
                    Vec3f meanColor(0, 0, 0);
                    for (const auto& sample : samples) {
                        meanColor[0] += (float) sample[0];
                        meanColor[1] += (float) sample[1];
                        meanColor[2] += (float) sample[2];
                    }

                    meanColor[0] /= (float) samples.size();
                    meanColor[1] /= (float) samples.size();
                    meanColor[2] /= (float) samples.size();

                    Vec3b bgrColor = {
                            static_cast<uchar>(meanColor[0]),
                            static_cast<uchar>(meanColor[1]),
                            static_cast<uchar>(meanColor[2])
                    };
                    Vec3b hsvColor = bgr2hsv(bgrColor);

                    Vec3b currentHsv = bgr2hsv(image.at<Vec3b>(y, x));

                    Vec3b newHsv;
                    newHsv[0] = hsvColor[0];

                    if (currentHsv[2] < 30) {
                        newHsv[1] = min(hsvColor[1], static_cast<uchar>(128));
                    } else if (currentHsv[2] > 220) {
                        newHsv[1] = min(hsvColor[1], static_cast<uchar>(128));
                    } else {
                        newHsv[1] = hsvColor[1];
                    }

                    if (currentHsv[2] < 30) {
                        newHsv[2] = min(static_cast<uchar>(100), hsvColor[2]);
                    } else if (currentHsv[2] > 220) {
                        newHsv[2] = max(static_cast<uchar>(150), hsvColor[2]);
                    } else {
                        newHsv[2] = currentHsv[2];
                    }

                    // Convert back to BGR
                    result.at<Vec3b>(y, x) = hsv2bgr(newHsv);
                }
            }
        }
    }

    return result;
}

Mat reduceNoise(const Mat& image, const Mat& noiseMask) {
    // Apply bilateral filter to the noisy regions only
    Mat result = image.clone();

    Mat denoisedRegion = customBilateralFilter(image, 5.0f, 30.0f);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (noiseMask.at<uchar>(y, x) == 0) {
                result.at<Vec3b>(y, x) = denoisedRegion.at<Vec3b>(y, x);
            }
        }
    }

    return result;
}


// 1) Inpainting (scratch + damage)
Mat doInpaint(const Mat& src) {
    Mat scratchMask = detectScratchLines(src);
    Mat damageMask  = detectDamageRegions(src);
    Mat combined    = Mat::zeros(src.size(), CV_8UC1);
    for(int y=0; y<src.rows; y++)
        for(int x=0; x<src.cols; x++)
            if (scratchMask.at<uchar>(y,x) || damageMask.at<uchar>(y,x))
                combined.at<uchar>(y,x) = 255;
    return customInpaint(src, combined, 5);
}

Mat doDenoise(const Mat& src) {
    Mat noiseMask = detectNoise(src);
    return reduceNoise(src, noiseMask);
}

Mat doColor(const Mat& src) {
    Mat colorMask = detectColorDegradation(src);
    return restoreColorDegradation(src, colorMask);
}

