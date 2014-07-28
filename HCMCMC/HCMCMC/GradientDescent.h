#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class GradientDescent {
private:
	float EPS;		// very small number

	float a;		// variance of w
	float theta;	// variance of x
	float e;		// step
	float rel;		// reliance

public:
	GradientDescent(void);
	float run(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, int niter);

private:
	float E(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q);
	cv::Mat dEx(cv::Mat& x, cv::Mat& w, cv::Mat& q);
	void dExi(int i, cv::Mat& x, cv::Mat& w, cv::Mat& q, cv::Mat& ret);
	cv::Mat dEw(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q);
	void dEwk(int k, cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, cv::Mat& ret);
};

