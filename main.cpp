#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/IOUtils.h"
#include "src/Detector.h"
#include "src/Restorer.h"

using namespace std;
using namespace cv;

int main() {

    Mat source = loadImage(R"(C:\Users\ruben\OneDrive\Desktop\School\Faculta\An3\Sem2\PI\Proiect\images\img3.bmp)");
    displayImage("Original Image", source);

    Mat lineMask = detectScratchLines(source);
    displayImage("Detected Scratch Lines Mask", lineMask);
//
//    Mat damaged = detectDamageRegions(source);
//    displayImage("Detected  Mask", damaged);
////
////    Mat noise = detectNoise(source);
////    displayImage("Detect noise", noise);
////
////    Mat color = detectColorDegradation(source);
////    displayImage("Detect color degradation", color);
//
//    Mat result = removeScratchLines(source, lineMask);
//    displayImage("Result", result);
//
//    Mat result1 = restoreDamagedRegions(source, damaged);
//    displayImage("result damaged", result1);

    //Mat result = restoreImage(source);
    //displayImage("Result", result);
    //displayComparison("After", source, result);

    waitKey();

    return 0;
}