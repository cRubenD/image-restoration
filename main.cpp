#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/IOUtils.h"
#include "src/Detector.h"
using namespace std;
using namespace cv;

int main() {

    Mat source = loadImage(R"(C:\Users\ruben\OneDrive\Desktop\School\Faculta\An3\Sem2\PI\Proiect\images\img20.bmp)");
    displayImage("Original Image", source);

    Mat lineMask = detectScratchLines(source);
    displayImage("Detected Scratch Lines Mask", lineMask);

    Mat line = detectDamageRegions(source);
    displayImage("Detected  Mask", line);

    Mat noise = detectNoise(source);
    displayImage("Detect noise", noise);

    Mat color = detectColorDegradation(source);
    displayImage("Detect color degradation", color);

    //displayComparison("After", source, lineMask);

    waitKey();

    return 0;
}