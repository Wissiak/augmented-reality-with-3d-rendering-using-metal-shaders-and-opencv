#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using std::cout;
using std::endl;

/*
https://thecodinginterface.com/blog/opencv-cpp-vscode/

mkdir build && cd build
cmake ..
cmake --build . --config Release
*/

class ARWebcam {
    public:
        ARWebcam() {
            
        }

    void video_in() {
        cv::Mat myImage;
        cv::namedWindow("Video Player");
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) { 
            cout << "No video stream detected" << endl;
            system("pause");
        }
        while (true) {
            cap >> myImage;
            if (myImage.empty()) {
            break;
            }
            imshow("Video Player", myImage);
            char c = (char) cv::waitKey(1); 

            // If 'Esc' is entered break the loop           
            if (c == 27) { 
            break;
            }
        }
        cap.release();
    }
};

int main(int argc, char **argv) {

    ARWebcam arWebcam;
    arWebcam.video_in();

    return 0;
}