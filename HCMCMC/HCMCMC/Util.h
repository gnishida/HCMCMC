#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class Util {
protected:
	Util();

public:
	static void normalize(cv::Mat& x);
	static float randu();
	static float randu(float a, float b);
	static void display(cv::Mat& m, int N);
	static void displayMat3f(cv::Mat& m, int N);
	static void displayVector(std::vector<int>& vec);
};

