#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/IOUtils.h"
#include "src/Restorer.h"

using namespace std;
using namespace cv;

int main() {

    cout << "Enter image number (e.g. 10): ";
    string filename;
    cin >> filename;

    string basePath = R"(C:\Users\ruben\OneDrive\Desktop\School\Faculta\An3\Sem2\PI\Proiect\images\img)";
    string filepath = basePath + filename + ".bmp";

    Mat source = loadImage(filepath);
    if (source.empty()) {
        cerr << "Failed to load " << filepath << "\n";
        return -1;
    }

    vector<pair<string, RestoreOp>> ops = {
            {"Remove scratches/damaged regions", doInpaint},
            {"Reduce noise",                doDenoise},
            {"Color correction",            doColor}
    };

    vector<RestoreOp> pipeline;  // Holds the user-selected steps

    while (true) {
        cout << "\nSelect a restoration operation (number), P to preview, or 0 to apply & exit:\n";
        for (int i = 0; i < (int)ops.size(); ++i)
            cout << "  " << (i+1) << ") " << ops[i].first << "\n";
        cout << "  P) Preview current pipeline\n";
        cout << "  0) Apply pipeline and finish\n";

        string input;
        cin >> input;

        if (input == "0") {
            break;
        }
        else if (input == "P" || input == "p") {
            Mat preview = source.clone();
            for (auto &op : pipeline)
                preview = op(preview);
            displayImage("Pipeline Preview", preview);
            waitKey(1);
        }
        else {
            int choice = stoi(input);
            if (choice >= 1 && choice <= (int)ops.size()) {
                pipeline.push_back(ops[choice-1].second);
                cout << "Added: \"" << ops[choice-1].first << "\"\n";
                Mat preview = source.clone();
                for (auto &op : pipeline)
                    preview = op(preview);
                displayImage("Preview After Addition", preview);
                waitKey(1);
            }
            else {
                cout << "Invalid option, please try again.\n";
            }
        }
    }

    Mat result = source.clone();
    for (auto &op : pipeline)
        result = op(result);
    displayImage("Final Restored Result", result);

    bool saveRequested = false;
    string savePath;

    while (true) {
        int key = waitKey(100); // Check every 100ms

        if (!saveRequested) {
            cout << "\nDo you want to save the restored image? (y/n): ";
            char saveChoice;
            cin >> saveChoice;
            saveRequested = true;

            if (saveChoice == 'y' || saveChoice == 'Y') {
                cout << "Enter a filename (without path), e.g. result1.bmp: ";
                string outFilename;
                cin >> outFilename;

                string saveDir = R"(C:\Users\ruben\OneDrive\Desktop\School\Faculta\An3\Sem2\PI\Proiect\saved_images\)";
                savePath = saveDir + outFilename;

                saveImage(savePath, result);
                cout << "Image saved to: " << savePath << endl;
                cout << "Press any key in the image window to exit." << endl;
            } else {
                cout << "Image not saved. Press any key in the image window to exit." << endl;
            }
        }

        if (key >= 0) {
            break;
        }
    }

    destroyAllWindows();
    return 0;
}