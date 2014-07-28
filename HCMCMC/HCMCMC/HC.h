#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class HC {
protected:
	HC() {}

public:
	static cv::Mat run(cv::Mat& xt, cv::Mat& zp, cv::Mat& wh);
};

