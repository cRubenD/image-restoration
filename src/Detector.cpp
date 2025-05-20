//
// Created by ruben on 4/29/2025.
//

#include "Detector.h"
#include "Helper.h"
#include "IOUtils.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

float sobelX[3][3] = { { -1,0,1 }, { -2,0,2 }, { -1,0,1 } };
float sobelY[3][3] = { { -1,-2,-1 }, { 0,0,0 }, { 1,2,1 } };

// calculeaza magnitudinea gradientului imaginii in fiecare punct
Mat applySobelFilter(const Mat& grayscaleImage) {

    Mat gradX(grayscaleImage.size(), CV_32F, Scalar(0));
    Mat gradY(grayscaleImage.size(), CV_32F, Scalar(0));

    for(int y = 1; y < grayscaleImage.rows - 1; y++) {
        for (int x = 1; x < grayscaleImage.cols - 1; x++) {
            float sumX = 0.0f;
            float sumY = 0.0f;

            // pentru fiecare pixel se face convolutia cu cele doua nuclee
            for(int ky = -1; ky <= 1; ky++) {
                for(int kx = -1; kx <= 1; kx++) {
                    float val = grayscaleImage.at<uchar>(y + ky, x + kx);
                    sumX += val * sobelX[ky + 1][kx + 1];
                    sumY += val * sobelY[ky + 1][kx + 1];
                }
            }

            gradX.at<float>(y, x) = sumX;
            gradY.at<float>(y, x) = sumY;
        }
    }

    // calculam magnitudinea gradientului
    Mat magnitude(grayscaleImage.rows, grayscaleImage.cols, CV_32F);
    for(int y = 0; y < grayscaleImage.rows; y++) {
        for (int x = 0; x < grayscaleImage.cols; x++) {
            float gx = gradX.at<float>(y, x);
            float gy = gradY.at<float>(y, x);
            magnitude.at<float>(y, x) = sqrt(gx * gx + gy * gy);
        }
    }

    double minVal, maxVal;
    minMaxLoc(magnitude, &minVal, &maxVal);

    Mat normalizedMagnitude(grayscaleImage.rows, grayscaleImage.cols, CV_8UC1);
    if(maxVal > minVal){

        // se scaleaza fiecare pixel a.i. cel mai mic gradient sa devina 0 iar cel mai mare 255
        for(int y = 0; y < grayscaleImage.rows; y++) {
            for (int x = 0; x < grayscaleImage.cols; x++) {
                normalizedMagnitude.at<uchar>(y, x) = static_cast<uchar> (255 * (magnitude.at<float>(y, x) - minVal) / (maxVal - minVal));
            }
        }
    }

    return normalizedMagnitude;
}

// detectam zgarieturi (linii)
Mat detectScratchLines(const Mat& image) {

    Mat grayscale = convertToGrayscale(image);
    //displayImage("gray", grayscale);
    Mat edges = applySobelFilter(grayscale);
    //displayImage("Sobel Magnitude", edges);
    Mat thresholdedEdges = applyThreshold(edges, 60);

    // manual Hough
    int width = thresholdedEdges.cols;
    int height = thresholdedEdges.rows;
    int diag = static_cast<int>(sqrt(width * width + height * height));
    int rhoMax = diag;
    int rhoRange = 2 * rhoMax;
    int thetaSteps = 180;
    vector<vector<int>> accumulator(rhoRange, vector<int>(thetaSteps, 0));
    float dTheta = CV_PI / thetaSteps;

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(thresholdedEdges.at<uchar>(y, x) == 255) {
                for(int t = 0; t < thetaSteps; t++) {
                    float theta = (float) t * dTheta;
                    int rho = cvRound((float) x * cos(theta) + (float) y * sin(theta)) + rhoMax;
                    if(rho >= 0 && rho < rhoRange) {
                        accumulator[rho][t]++;
                    }
                }
            }
        }
    }

    int voteThresh = 100;
    vector<pair<int,int>> peaks;
    for(int r=0; r<rhoRange; r++) {
        for(int t=0; t<thetaSteps; t++) {
            if(accumulator[r][t] > voteThresh) {
                peaks.emplace_back(r, t);
            }
        }
    }

    Mat mask = Mat::zeros(height, width, CV_8UC1);
    for(pair<int, int> p : peaks) {
        int r = p.first - rhoMax;
        float theta = (float) p.second * dTheta;
        float a = cos(theta), b = sin(theta);
        float x0 = a * (float) r, y0 = b * (float) r;
        Point pt1, pt2;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(mask, pt1, pt2, Scalar(255), 2);
    }

    return mask;
}

// calculam media si deviatia standard pt pixelul curent in jurul unei ferestre pe cele 3 canale
pair<Scalar, Scalar> calculateLocalStatistics(const Mat& image, int x, int y, int windowSize) {
    int halfSize = windowSize / 2;
    int count = 0;
    Scalar mean(0, 0, 0);
    Scalar stdDev;
    Scalar sumSquares(0, 0, 0);

    for (int dy = -halfSize; dy <= halfSize; dy++) {
        for (int dx = -halfSize; dx <= halfSize; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
                Vec3b pixel = image.at<Vec3b>(ny, nx);
                for (int c = 0; c < 3; c++) {
                    mean[c] += pixel[c];
                    sumSquares[c] += pixel[c] * pixel[c];
                }
                count++;
            }
        }
    }

    if (count > 1) { // Minim 2 pixeli pentru deviație
        for (int c = 0; c < 3; c++) {
            mean[c] /= count;
            double variance = (sumSquares[c] - mean[c] * mean[c] * count) / (count - 1);
            stdDev[c] = sqrt(variance);
        }
    }

    return {mean, stdDev};
}

Mat detectDamageRegions(const Mat& image) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    const int windowSize = 11;
    const double thresholdDeviation = 2.0;

    // pt fiecare pixel central, se compara abaterea fata de media locala cu un multiplu
    // al deviatiei standard
    for(int y = windowSize / 2; y < image.rows - windowSize / 2; y++) {
        for(int x = windowSize / 2; x < image.cols - windowSize / 2; x++) {
            Scalar mean, stdDev;
            tie(mean, stdDev) = calculateLocalStatistics(image, x, y, windowSize);

            Vec3b currentPixel = image.at<Vec3b>(y, x);

            bool isDamaged = false;
            for(int c = 0; c < 3; c++) {
                double difference = abs(currentPixel[c] - mean[c]);
                if(difference > thresholdDeviation * stdDev[c] && stdDev[c] > 3.0) {
                    isDamaged = true;
                    break;
                }
            }

            if(isDamaged) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    // dilatarea + eroziunea imaginii pe o masca de 3 x 3
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat closedMask;
    // inchide golurile si uneste regiunile apropiate de masca initiala
    morphologyEx(mask, closedMask, MORPH_CLOSE, kernel);

    return closedMask;
}

void morphOpen(const Mat& src, Mat& dst, const Mat& kernel) {
    Mat eroded(src.size(), src.type());
    morphErode(src, eroded, kernel);
    morphDilate(eroded, dst, kernel);
}

void morphClose(const Mat& src, Mat& dst, const Mat& kernel) {
    Mat dilated(src.size(), src.type());
    morphDilate(src, dilated, kernel);
    morphErode(dilated, dst, kernel);
}

Mat detectNoise(const Mat& image) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    Mat smoothed(image.size(), image.type());

    // creeare kernel gaussian si filtrare
    const int kernelSize = 7;
    const float sigma = 2.0f;
    vector<vector<float>> gaussianKernel = makeGaussianKernel(kernelSize, sigma);
    int halfSize = kernelSize / 2;

    for (int y = halfSize; y < image.rows - halfSize; y++) {
        for (int x = halfSize; x < image.cols - halfSize; x++) {
            Vec3f pixel_sum(0, 0, 0);

            // convolutia manuala gaussiana
            for (int ky = -halfSize; ky <= halfSize; ky++) {
                for (int kx = -halfSize; kx <= halfSize; kx++) {
                    Vec3b pixel = image.at<Vec3b>(y + ky, x + kx);
                    float weight = gaussianKernel[ky + halfSize][kx + halfSize];

                    pixel_sum[0] += (float) pixel[0] * weight;
                    pixel_sum[1] += (float) pixel[1] * weight;
                    pixel_sum[2] += (float) pixel[2] * weight;
                }
            }
            smoothed.at<Vec3b>(y, x) = Vec3b(
                    static_cast<uchar>(pixel_sum[0]),
                    static_cast<uchar>(pixel_sum[1]),
                    static_cast<uchar>(pixel_sum[2])
            );
        }
    }

    Mat difference(image.size(), CV_8UC3);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b original = image.at<Vec3b>(y, x);
            Vec3b smooth = smoothed.at<Vec3b>(y, x);

            // calcularea diferentei patratice pt contrast mai bun
            difference.at<Vec3b>(y, x) = Vec3b(
                    min(255, (int)pow(abs(original[0] - smooth[0]), 2) / 16),
                    min(255, (int)pow(abs(original[1] - smooth[1]), 2) / 16),
                    min(255, (int)pow(abs(original[2] - smooth[2]), 2) / 16)
            );
        }
    }

    Mat grayDifference = convertToGrayscale(difference);
    Mat adaptiveThreshold = adaptiveNoiseThreshold(grayDifference, 9, 1.0);

    Mat kernel = Mat::ones(2, 2, CV_8UC1);
    Mat openedMask, closedMask;

    morphOpen(adaptiveThreshold, openedMask, kernel);

    morphClose(openedMask, closedMask, kernel);

    // rezultatul final e o masca binara, care indica exact pixelii afectati de zgomot
    return closedMask;
}

Mat detectColorDegradation(const Mat& image) {
    int rows = image.rows, cols = image.cols;
    Mat mask(rows, cols, CV_8UC1, Scalar(0));

    // detectam pixelii cu degradare de culoare si ii setam la alb
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto& bgr = image.at<Vec3b>(y,x);
            Vec3b hsv = bgr2hsv(bgr);
            uchar H = hsv[0], S = hsv[1], V = hsv[2];

            if (S < 30 || V < 30 || V > 220) {
                mask.at<uchar>(y,x) = 255;
            }
        }
    }

    // dilatare manuala cu un patrat de 2k + 1
    int k = 2;
    Mat dil(rows, cols, CV_8UC1, Scalar(0));
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (mask.at<uchar>(y,x) == 255) {
                for (int dy = -k; dy <= k; dy++) {
                    for (int dx = -k; dx <= k; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols)
                            dil.at<uchar>(yy,xx) = 255;
                    }
                }
            }
        }
    }

    // dilatare + eroziune manuala inchidem golurile interne
    Mat closed(rows, cols, CV_8UC1, Scalar(0));
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            bool keep = true;
            for (int dy = -k; dy <= k && keep; dy++) {
                for (int dx = -k; dx <= k; dx++) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0||yy >= rows || xx < 0 || xx >= cols) continue;
                    if (dil.at<uchar>(yy,xx) == 0) {
                        keep = false;
                        break;
                    }
                }
            }
            if (keep) closed.at<uchar>(y,x) = 255;
        }
    }

    return closed;
}


