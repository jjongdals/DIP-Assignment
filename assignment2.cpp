#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void wienerFilter(const Mat& inputImg, const Mat& kernelImg, float noisePower, Mat& outputImg) {
    //dft 크기 계산 => 빠르게 퓨리에 변환을 하기 위해서 optimal한 크기로 계산
    int m = getOptimalDFTSize(inputImg.rows);
    int n = getOptimalDFTSize(inputImg.cols);
    
    Mat paddedInput, paddedKernel, complexInput, complexKernel;
    // 이미지 패딩 추가, 패딩 영역은 0을 넣음으로써 검은색으로
    copyMakeBorder(inputImg, paddedInput, 0, m - inputImg.rows, 0, n - inputImg.cols, BORDER_CONSTANT, Scalar::all(0));
    copyMakeBorder(kernelImg, paddedKernel, 0, m - kernelImg.rows, 0, n - kernelImg.cols, BORDER_CONSTANT, Scalar::all(0));
    //퓨리에 변환 로직 구성
    Mat planesInput[] = {Mat_<float>(paddedInput), Mat::zeros(paddedInput.size(), CV_32F)}; //두 mat 객체의 배열 존재 1번쨰는 float type, 2번째는 인풋 이미지와 동일한 영행렬
    merge(planesInput, 2, complexInput); // 복소수 행렬로 합침
    dft(complexInput, complexInput); // 퓨리에 변환

    Mat planesKernel[] = {Mat_<float>(paddedKernel), Mat::zeros(paddedKernel.size(), CV_32F)}; //커널에 대해서 ㅇㅇ, 위랑 같음
    merge(planesKernel, 2, complexKernel);
    dft(complexKernel, complexKernel);
    
    //wiener필터 로직 구현
    Mat complexH(complexInput.size(), complexInput.type()); // 복소수 이미지 초기화
    /*
     복소수 이미지에 대해 반복문으로 직접 접근
     1) 행에 대해 0부터 size - 1만큼
     2) 열에 대해서도 마찬가지로
     */
    for (int y = 0; y < complexInput.rows; y++) {
        for (int x = 0; x < complexInput.cols; x++) {
            // 복소수의 실수랑 허수 정의, (y, x) 의 위치로부터 값들을 읽어들임
            Vec2f I = complexInput.at<Vec2f>(y, x);
            Vec2f K = complexKernel.at<Vec2f>(y, x);
            // 실수랑 허수 제곱의 합 => 복소수는 제곱을 해야 나타낼 수 있음
            float powerK = K[0] * K[0] + K[1] * K[1];
            // 분모로 나타내기 위해서, powerK랑 인풋으로 들어오는 k값의 합
            float denominator = powerK + noisePower;
            // 분모로 0이 들어가면 연산이 안되니까 이에 대한 예외처리 진행 0이 아니면 나눠주고
            if (denominator != 0) {
                complexH.at<Vec2f>(y, x)[0] = (I[0] * K[0] + I[1] * K[1]) / denominator;
                complexH.at<Vec2f>(y, x)[1] = (I[1] * K[0] - I[0] * K[1]) / denominator;
            }
            // 0이 되면 연산 x
            else {
                complexH.at<Vec2f>(y, x) = Vec2f(0, 0);
            }
        }
    }

    idft(complexH, outputImg, DFT_SCALE | DFT_REAL_OUTPUT);
    outputImg = outputImg(Rect(0, 0, inputImg.cols, inputImg.rows));
}



int main() {
    
    Mat img1 = imread("/Users/parkjongwon/Desktop/Wiener_input1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("/Users/parkjongwon/Desktop/Wiener_input2.png", IMREAD_GRAYSCALE);
    Mat kernelImg = imread("/Users/parkjongwon/Desktop/Wiener_kernel.png", IMREAD_GRAYSCALE);
    
    
    // normalize 진행
    img1.convertTo(img1, CV_32F, 1.0/255);
    img2.convertTo(img2, CV_32F, 1.0/255);
    kernelImg.convertTo(kernelImg, CV_32F, 1.0/255);
    kernelImg /= sum(kernelImg);
    
    Mat output1, output2;
    wienerFilter(img1, kernelImg, 0.001, output1);
    wienerFilter(img2, kernelImg, 0.0017, output2);
    // 원본과 비교를 위해서
    imshow("Origin Image1", img1);
    imshow("Origin Image2", img2);
    // 다음 이미지는 결과 이미지
    imshow("Restored Image1", output1);
    imshow("Restored Image2", output2);
    waitKey(0);
    
    return 0;
}

