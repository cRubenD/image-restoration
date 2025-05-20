//
// Created by ruben on 5/20/2025.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// Function to detect scratches using edge detection and Hough transform
Mat detectScratchesOpenCV(const Mat& src) {
    Mat gray;

    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    Mat edges;
    Canny(blurred, edges, 50, 150);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(edges, edges, kernel);

    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, 30, 10);

    Mat scratchMask = Mat::zeros(src.size(), CV_8UC1);
    for (size_t i = 0; i < lines.size(); i++) {
        line(scratchMask, Point(lines[i][0], lines[i][1]),
             Point(lines[i][2], lines[i][3]), Scalar(255), 2);
    }

    dilate(scratchMask, scratchMask, kernel);

    return scratchMask;
}

// Function to detect damaged regions using adaptive thresholding
Mat detectDamageRegionsOpenCV(const Mat& src) {
    Mat gray;

    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    Mat binary;
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                      THRESH_BINARY_INV, 11, 2);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(binary, labels, stats, centroids);

    Mat damageMask = Mat::zeros(src.size(), CV_8UC1);

    int minArea = 50;
    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area > minArea && area < 1000) {
            Mat componentMask = (labels == i);
            bitwise_or(damageMask, componentMask, damageMask);
        }
    }

    return damageMask;
}

// Function to detect noise using local variance
Mat detectNoiseOpenCV(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    Mat mean, variance;
    meanStdDev(gray, mean, variance);
    double overallStdDev = variance.at<double>(0);

    Mat localVariance;
    Mat meanImg;
    blur(gray, meanImg, Size(5, 5));

    Mat meanSquare;
    blur(gray.mul(gray), meanSquare, Size(5, 5));

    localVariance = meanSquare - meanImg.mul(meanImg);

    Mat noiseMask;
    double thresholdValue = overallStdDev * 1.5;
    threshold(localVariance, noiseMask, thresholdValue, 255, THRESH_BINARY);

    noiseMask.convertTo(noiseMask, CV_8UC1);

    return noiseMask;
}

// Function to detect color degradation
Mat detectColorDegradationOpenCV(const Mat& src) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);

    Mat lowSatMask, lowValMask, highValMask;
    threshold(hsvChannels[1], lowSatMask, 30, 255, THRESH_BINARY_INV);  // S < 30
    threshold(hsvChannels[2], lowValMask, 30, 255, THRESH_BINARY_INV);  // V < 30
    threshold(hsvChannels[2], highValMask, 220, 255, THRESH_BINARY);    // V > 220

    Mat colorDegradationMask;
    bitwise_or(lowSatMask, lowValMask, colorDegradationMask);
    bitwise_or(colorDegradationMask, highValMask, colorDegradationMask);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    dilate(colorDegradationMask, colorDegradationMask, kernel);
    morphologyEx(colorDegradationMask, colorDegradationMask, MORPH_CLOSE, kernel);

    return colorDegradationMask;
}

// Inpainting function
Mat doInpaintWithOpenCV(const Mat& src) {
    Mat scratchMask = detectScratchesOpenCV(src);
    Mat damageMask = detectDamageRegionsOpenCV(src);

    Mat mask;
    bitwise_or(scratchMask, damageMask, mask);

    imshow("Inpainting Mask", mask);

    Mat result;
    inpaint(src, mask, result, 5, INPAINT_TELEA);  // or INPAINT_NS

    return result;
}

// Denoising function
Mat doDenoiseWithOpenCV(const Mat& src) {
    Mat result;

    if (src.channels() == 3) {
        fastNlMeansDenoisingColored(src, result, 10, 10, 7, 21);
    } else {
        fastNlMeansDenoising(src, result, 10, 7, 21);
    }

    return result;
}

// Color correction function
Mat doColorCorrectionWithOpenCV(const Mat& src) {
    if (src.channels() != 3) {
        return src.clone();
    }

    Mat result = src.clone();
    vector<Mat> bgr;
    split(result, bgr);

    Scalar means = mean(result);
    double avgMean = (means[0] + means[1] + means[2]) / 3.0;

    for (int i = 0; i < 3; i++) {
        bgr[i] = bgr[i] * (avgMean / means[i]);
    }

    merge(bgr, result);

    Mat lab;
    cvtColor(result, lab, COLOR_BGR2Lab);

    vector<Mat> labChannels;
    split(lab, labChannels);

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(labChannels[0], labChannels[0]);

    merge(labChannels, lab);
    cvtColor(lab, result, COLOR_Lab2BGR);

    return result;
}

// Function to improve the image by applying all restoration steps
Mat restoreImage(const Mat& src) {
    Mat inpainted = doInpaintWithOpenCV(src);

    Mat denoised = doDenoiseWithOpenCV(inpainted);

    Mat colorCorrected = doColorCorrectionWithOpenCV(denoised);

    return colorCorrected;
}

int main() {
    cout << "Enter image number (e.g. 10): ";
    string filename;
    cin >> filename;

    string basePath = R"(C:\Users\ruben\OneDrive\Desktop\School\Faculta\An3\Sem2\PI\Proiect\images\img)";
    string filepath = basePath + filename + ".bmp";

    Mat source = imread(filepath);
    if (source.empty()) {
        cerr << "Failed to load " << filepath << "\n";
        return -1;
    }

    // Display the original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", source);

    // Menu loop
    int choice = 0;
    do {
        cout << "\nSelect an operation:\n";
        cout << "1. Remove scratches/damaged regions\n";
        cout << "2. Reduce noise\n";
        cout << "3. Color correction\n";
        cout << "4. Full restoration (all steps)\n";
        cout << "0. Exit\n";
        cout << "Your choice: ";
        cin >> choice;

        Mat result;
        string windowTitle;

        switch (choice) {
            case 1:
                result = doInpaintWithOpenCV(source);
                windowTitle = "After Inpainting";
                break;
            case 2:
                result = doDenoiseWithOpenCV(source);
                windowTitle = "After Denoising";
                break;
            case 3:
                result = doColorCorrectionWithOpenCV(source);
                windowTitle = "After Color Correction";
                break;
            case 4:
                result = restoreImage(source);
                windowTitle = "Full Restoration";
                break;
            case 0:
                cout << "Exiting...\n";
                break;
            default:
                cout << "Invalid choice, please try again.\n";
                continue;
        }

        if (choice >= 1 && choice <= 4) {
            namedWindow(windowTitle, WINDOW_AUTOSIZE);
            imshow(windowTitle, result);

            // Ask if user wants to save the result
            cout << "Do you want to save this result? (y/n): ";
            char saveChoice;
            cin >> saveChoice;

            if (saveChoice == 'y' || saveChoice == 'Y') {
                string saveDir = R"(C:\Users\ruben\OneDrive\Desktop\School\Faculta\An3\Sem2\PI\Proiect\saved_images\)";
                string outFilename = "result_" + to_string(choice) + "_" + filename + ".bmp";
                string savePath = saveDir + outFilename;

                bool success = imwrite(savePath, result);
                if (success) {
                    cout << "Image saved to: " << savePath << endl;
                } else {
                    cerr << "Failed to save image!" << endl;
                }
            }
        }

    } while (choice != 0);

    waitKey(0);
    destroyAllWindows();
    return 0;
}