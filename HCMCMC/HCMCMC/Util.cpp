#include "Util.h"

#ifndef SQR
#define SQR(x)		((x) * (x))
#endif

/**
 * Normalize matrix such that each column has a mean 0 and a variance 1.
 */
void Util::normalize(cv::Mat& x) {
	cv::Mat avg;
	cv::reduce(x, avg, 0, CV_REDUCE_AVG);

	for (int c = 0; c < x.cols; ++c) {
		float total = 0.0f;
		for (int r = 0; r < x.rows; ++r) {
			total += SQR(x.at<float>(r, c) - avg.at<float>(0, c));
		}
		float stddev = sqrt(total / (float)x.rows);
		
		for (int r = 0; r < x.rows; ++r) {
			x.at<float>(r, c) = (x.at<float>(r, c) - avg.at<float>(0, c)) / stddev;
		}
	}
}