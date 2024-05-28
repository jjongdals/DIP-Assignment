
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

cv::Mat alphaTrimmedMean(const cv::Mat& src, cv::Size sz, float alpha) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    int dx = sz.width / 2;
    int dy = sz.height / 2;
    int endIndex, startIndex, size, removeCount;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            std::vector<float> neighborhood;

            for (int ny = -dy; ny <= dy; ny++) {
                for (int nx = -dx; nx <= dx; nx++) {
                    int posX = x + nx;
                    int posY = y + ny;

                    // 패딩을 적용하지 않고, 범위를 벗어난 경우 무시
                    if (posX >= 0 && posX < src.cols && posY >= 0 && posY < src.rows) {
                        neighborhood.push_back(src.at<float>(posY, posX));
                    }
                }
            }
            //sort해줌
            std::sort(neighborhood.begin(), neighborhood.end());
            
            size = neighborhood.size();
            removeCount = static_cast<int>(size * alpha * 0.5);
            startIndex = removeCount;
            endIndex = size - removeCount;
            float sum = 0;

            for (int i = startIndex; i < endIndex; i++) {
                sum += neighborhood[i];
            }
            dst.at<float>(y, x) = sum / (endIndex - startIndex);
        }
    }

    // 결과 이미지의 픽셀 값을 0에서 255로 클리핑
    cv::Mat resultDst;
    dst.convertTo(resultDst, CV_8U, 1.0, 0.0);  // 데이터 타입을 CV_8U로 변환하면서 스케일 조정

    return resultDst;
}

int main() {
    //경로는 내껄로 ㅇㅇ 수정 필요
    cv::Mat img = cv::imread("/Users/parkjongwon/Desktop/noisy.png", cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32FC1); // float 형식으로 변환

    cv::Mat filtered = alphaTrimmedMean(img, cv::Size(3, 3), 0.18);

    cv::imshow("alpha-trimmed filter", filtered);
    cv::waitKey(0);

    return 0;
}
