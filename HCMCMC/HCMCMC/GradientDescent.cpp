#include "GradientDescent.h"
#include "Util.h"

GradientDescent::GradientDescent() {
	EPS = 0.0000001f;
	a = 1.0f;
	theta = 1.0f;
	e = 0.01f;
	rel = 10.0f;
}

/**
 * Gradient descent
 */
float GradientDescent::run(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, int niter) {
	int cnt = 0;
	for (int i = 0; i < niter; ++i, ++cnt) {
		float old_E = E(x, w, wh, q);
		x -= e * dEx(x, w, q);
		Util::normalize(x);

		//w -= e * dEw(x, w, wh, q);

		if (old_E > E(x, w, wh, q) && old_E - E(x, w, wh, q) < EPS) break;
	}

	return E(x, w, wh, q);
}

/**
 * Objective function E() defined in eq(4).
 */
float GradientDescent::E(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q) {
	float ret = cv::sum((w - wh).mul(w - wh))[0] * 0.5f / a / a;

	ret += cv::sum(x.mul(x))[0] * 0.5f / theta / theta;
	for (int i = 0; i < x.rows; ++i) {
		for (int j = i+1; j < x.rows; ++j) {
			for (int k = 0; k < w.rows; ++k) {
				float dot = rel * w.row(k).dot(x.row(j) - x.row(i));
				ret -= (1.0f - q.at<float>(i, j, k)) * dot;
				ret += logf(1.0f + expf(dot));
			}
		}
	}

	return ret;
}

cv::Mat GradientDescent::dEx(cv::Mat& x, cv::Mat& w, cv::Mat& q) {
	cv::Mat ret = cv::Mat::zeros(x.rows, x.cols, CV_32F);
	for (int i = 0; i < ret.rows; ++i) {
		cv::Mat row_i = ret.colRange(0, ret.cols).rowRange(i, i+1);
		dExi(i, x, w, q, row_i);
	}

	return ret;
}

void GradientDescent::dExi(int i, cv::Mat& x, cv::Mat& w, cv::Mat& q, cv::Mat& ret) {
	x.row(i).copyTo(ret);
	ret /= (theta * theta);
	for (int j = 0; j < x.rows; ++j) {
		if (j == i) continue;
		for (int k = 0; k < w.rows; ++k) {
			ret -= w.row(k) * q.at<float>(i, j, k);
		}
	}
	for (int j = i + 1; j < x.rows; ++j) {
		for (int k = 0; k < w.rows; ++k) {
			ret += w.row(k);
		}
	}
	for (int j = i + 1; j < x.rows; ++j) {
		for (int k = 0; k < w.rows; ++k) {
			float ex = expf(rel * w.row(k).dot(x.row(j) - x.row(i)));
			ret -= w.row(k) * ex / (1.0f + ex);
		}
	}
	for (int j = 0; j < i; ++j) {
		for (int k = 0; k < w.rows; ++k) {
			float ex = expf(rel * w.row(k).dot(x.row(i) - x.row(j)));
			ret += w.row(k) * ex / (1.0f + ex);
		}
	}
}

cv::Mat GradientDescent::dEw(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q) {
	cv::Mat ret = cv::Mat::zeros(w.rows, w.cols, CV_32F);
	for (int k = 0; k < ret.rows; ++k) {
		cv::Mat row_k = ret.colRange(0, ret.cols).rowRange(k, k+1);
		dEwk(k, x, w, wh, q, row_k);
	}

	return ret;
}

void GradientDescent::dEwk(int k, cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, cv::Mat& ret) {
	((cv::Mat)((w.row(k) - wh.row(k)) / a / a)).copyTo(ret);
	for (int i = 0; i < x.rows; ++i) {
		for (int j = i + 1; j < x.rows; ++j) {
			ret -= (1 - q.at<float>(i, j, k)) * (x.row(j) - x.row(i));
		}
	}

	for (int i = 0; i < x.rows; ++i) {
		for (int j = i + 1; j < x.rows; ++j) {
			float ex = expf(rel * w.row(k).dot(x.row(j) - x.row(i)));
			ret += (x.row(j) - x.row(i)) * ex / (1.0f + ex);
		}
	}
}
