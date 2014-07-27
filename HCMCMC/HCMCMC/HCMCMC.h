#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class HCMCMC {
private:
	int N;	// num of samples
	int M;	// num of raters
	int S;	// num of characteristics of each sample
	int D;	// dimension of states
	int T;	// num of MCMC steps
	cv::Mat ws; // desired weight
	std::vector<cv::Mat> img;

public:
	HCMCMC(int N, int M, int S, int D, int T, cv::Mat& ws);
	void run();

private:
	cv::Mat grad_desc_test(cv::Mat& wh, cv::Mat& zp);
	int choose_next(cv::Mat& p);
	void save(cv::Mat& result, char* filename);
	float KStest(cv::Mat& result);
	void check_estimation(cv::Mat& est, cv::Mat& wh, cv::Mat& q);
};

