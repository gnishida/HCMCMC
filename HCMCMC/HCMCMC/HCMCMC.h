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
	void setupGroundTruth(std::vector<int>& size);
	int choose_next(cv::Mat& p);
	int choose_best(cv::Mat& p);
	cv::Mat getTrueValue(cv::Mat& zp);
	void save(cv::Mat result, int d1, int d2, char* filename);
	//float KStest(cv::Mat& result);
	float topN(cv::Mat& result, cv::Mat& truth, float topRatio);
	//void check_estimation(cv::Mat& x, cv::Mat& xt);
};

