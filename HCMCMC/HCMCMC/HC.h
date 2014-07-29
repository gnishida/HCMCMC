#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class HC {
protected:
	HC() {}

public:
	static cv::Mat run(std::vector<cv::Mat>& img, cv::Mat& zp, cv::Mat& wh, cv::Mat& xt, bool noise = true);
};

