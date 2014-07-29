#include "Util.h"
#include <stdlib.h>
#include "Permutation.h"

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
			x.at<float>(r, c) -= avg.at<float>(0, c);
			if (stddev > 0.0f) {
				x.at<float>(r, c) /= stddev;
			}
		}
	}
}

float Util::randu() {
	return randu(0.0f, 1.0f);
}

float Util::randu(float a, float b) {
	float r = (float)(rand() % 1000) / 1000.0f;//RAND_MAX;

	return a + (b - a) * r;
}

void Util::display(cv::Mat& m, int N) {
	Permutation perm(m.dims, N - 1);
	while (true) {
		std::vector<int> point(m.dims);
		for (int i = 0; i < m.dims; ++i) {
			point[i] = perm.data[i];
		}
		std::cout << m.at<float>(point.data()) << std::endl;

		if (!perm.next()) break;
	}
}

void Util::displayMat3f(cv::Mat& m, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				std::cout << m.at<float>(i, j, k) << std::endl;
			}
		}
	}
}

void Util::displayVector(std::vector<int>& vec) {
	std::cout << "[";
	for (int i = 0; i < vec.size(); ++i) {
		std::cout << vec[i];
		if (i < vec.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "]" << std::endl;
}
