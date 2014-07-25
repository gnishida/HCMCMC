#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class GibbsSampling {
private:
	int N;	// num of samples
	int M;	// num of raters
	int S;	// num of characteristics of each sample
	int D;	// dimension of states
	int T;	// num of MCMC steps
	float r;	// reliance
	float a;	// alpha
	float e;	// epsilon
	float theta;
	cv::Mat ws; // desired weight
	cv::Mat result;
	std::vector<cv::Mat> img;

public:
	GibbsSampling(int N, int M, int S, int D, int T, cv::Mat& ws);
	void run();

private:
	int grad_desc(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, int niter);
	cv::Mat grad_desc_test(cv::Mat& wh, cv::Mat& zp, int niter);
	int choose_next(cv::Mat& p);
	void normalize(cv::Mat& x);

	float E(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q);
	cv::Mat dEx(cv::Mat& x, cv::Mat& w, cv::Mat& q);
	void dExi(int i, cv::Mat& x, cv::Mat& w, cv::Mat& q, cv::Mat& ret);
	cv::Mat dEw(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q);
	void dEwk(int k, cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, cv::Mat& ret);
};

