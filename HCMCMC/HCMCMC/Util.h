#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class Util {
protected:
	Util();

public:
	static void normalize(cv::Mat& x);
};

