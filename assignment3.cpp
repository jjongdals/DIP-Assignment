#include <opencv2/opencv.hpp>

using namespace cv;

Mat extractSkeleton(const Mat& inputImage) {
    Mat img;
    threshold(inputImage, img, 127, 255, THRESH_BINARY_INV); // 간단한 이진화

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    Mat skeleton = Mat::zeros(img.size(), CV_8UC1);
    Mat temp, eroded;

    morphologyEx(img, temp, MORPH_OPEN, element);
    subtract(img, temp, temp);
    bitwise_or(skeleton, temp, skeleton);

    Mat prevImg = img.clone();

    while (true) {
        erode(img, eroded, element);
        morphologyEx(eroded, temp, MORPH_OPEN, element);
        subtract(eroded, temp, temp);
        bitwise_or(skeleton, temp, skeleton);

        if (countNonZero(img - eroded) == 0) {
            break;
        }
        eroded.copyTo(img);
    }
    return skeleton;
}

int main() {
    Mat img1 = imread("/Users/parkjongwon/Desktop/sk1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("/Users/parkjongwon/Desktop/sk2.png", IMREAD_GRAYSCALE);
    Mat img3 = imread("/Users/parkjongwon/Desktop/sk3.png", IMREAD_GRAYSCALE);

    Mat skeleton1 = extractSkeleton(img1);
    Mat skeleton2 = extractSkeleton(img2);
    Mat skeleton3 = extractSkeleton(img3);

    imshow("original1", img1);
    imshow("original2", img2);
    imshow("original3", img3);
    
    imshow("Skeleton1", skeleton1);
    imshow("Skeleton2", skeleton2);
    imshow("Skeleton3", skeleton3);
    
    imwrite("/Users/parkjongwon/Desktop/skeleton1.png", skeleton1);
    imwrite("/Users/parkjongwon/Desktop/skeleton2.png", skeleton2);
    imwrite("/Users/parkjongwon/Desktop/skeleton3.png", skeleton3);

    waitKey(0);

    return 0;
}
