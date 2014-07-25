#include "GibbsSampling.h"
#include <vector>
#include <random>

#ifndef SQR
#define SQR(x)		((x) * (x))
#endif

#define EPS		0.0000001

GibbsSampling::GibbsSampling(int N, int M, int S, int D, int T, cv::Mat& ws) : N(N), M(M), S(S), D(D), T(T), ws(ws) {
	r = 10.0f;
	a = 1.0f;
	theta = 1.0f;
	e = 0.01f;
}

void GibbsSampling::run() {
	// setup random variables
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> randu(0.0, 1.0);

	// setup images (which will be used as probability distribution)
	img.clear();
	for (int s = 0; s < S; ++s) {
		char filename[256];
		sprintf(filename, "truth%d.bmp", s);
		img.push_back(cv::imread(filename, 0));
	}

	// clear the result
	result = cv::Mat(img[0].rows, img[0].cols, CV_32F, cv::Scalar(0.0f));

	// initialize the state (center)
	std::vector<int> z;
	for (int d = 0; d < D; ++d) {
		z.push_back(img[0].rows * 0.5f);
	}

	// synthesize wh
	cv::Mat wh(M, S, CV_32F);
	cv::randu(wh, cv::Scalar(0.0f), cv::Scalar(1.0f));
	cv::Mat wh_col;
	cv::reduce(wh, wh_col, 1, CV_REDUCE_SUM);
	for (int k = 0; k < M; ++k) {
		for (int s = 0; s < S; ++s) {
			wh.at<float>(k, s) /= wh_col.at<float>(k, 0);
		}
	}
	std::cout << wh << std::endl;

	// MCMC
	for (int t = 0; t < T; ++t) {
		for (int d = 0; d < D; ++d) {
			// record the current state
			result.at<float>(z[0], z[1]) += 1;

			// sample N data
			cv::Mat zp(N, D, CV_32F);
			for (int i = 0; i < N; ++i) {
				for (int d2 = 0; d2 < D; ++d2) {
					if (d2 == d) {
						zp.at<float>(i, d2) = (int)(randu(mt) * img[0].rows);
					} else {
						zp.at<float>(i, d2) = z[d2];
					}
				}
			}

			// find the optimum by gradient descent
			std::vector<float> p = grad_desc_test(wh, zp, 100);

			// choose the next state according to the conditional distribution

		}
	}
}

/**
 * Gradient descent
 */
cv::Mat GibbsSampling::grad_desc_test(cv::Mat& wh, cv::Mat& zp, int niter) {
	// setup random variables
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> randu(0.0, 1.0);

	// given the N samples, compute the true scores
	cv::Mat xt = cv::Mat::zeros(N, S, CV_32F);
	for (int i = 0; i < N; ++i) {
		for (int s = 0; s < S; ++s) {
			xt.at<float>(i, s) = img[s].at<uchar>((int)zp.at<float>(i, 0), (int)zp.at<float>(i, 1));
		}
	}
	normalize(xt);

	std::cout << xt << std::endl;

	// TODO
	// We should add Gaussian to xt before synthesize q

	// synthesize q
	int sizes[3];
	sizes[0] = N;
	sizes[1] = N;
	sizes[2] = M;
	cv::Mat q = cv::Mat(3, sizes, CV_32F);
	q *= 0.0f;
	for (int i = 0; i < N; ++i) {
		for (int j = i + 1; j < N; ++j) {
			for (int k = 0; k < M; ++k) {
				int v1[3];
				v1[0] = i;
				v1[1] = j;
				v1[2] = k;
				int v2[3];
				v1[0] = j;
				v1[1] = i;
				v1[2] = k;

				if (1.0f / (1.0f + expf(r * wh.row(k).dot(xt.row(j) - xt.row(i)))) >= randu(mt)) {
					q.at<float>(cv::Vec3i(i, j, k)) = 1.0f;
				}
				q.at<float>(cv::Vec3i(j, i, k)) = 1.0f - q.at<float>(cv::Vec3i(i, j, k));
			}
		}
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%f, " , q.at<float>(cv::Vec3i(i, j, 0)));
		}
		printf("\n");
	}

	// initialize x, w
	cv::Mat x = cv::Mat::zeros(xt.rows, xt.cols, CV_32F);
	cv::Mat w(wh.rows, wh.cols, CV_32F);
	wh.copyTo(w);

	grad_desc(x, w, wh, q, niter);

	return x;
}

/**
 * Gradient descent
 */
int GibbsSampling::grad_desc(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, int niter) {
	int cnt = 0;
	for (int i = 0; i < niter; ++i, ++cnt) {
		float old_E = E(x, w, wh, q);
		x -= e * dEx(x, w, q);
		normalize(x);
		std::cout << x << std::endl;

		w -= e * dEw(x, w, wh, q);
		std::cout << w << std::endl;

		if (old_E > E(x, w, wh, q) && old_E - E(x, w, wh, q) < EPS) break;
	}

	return cnt;
}

/**
 * Choose the next state
 */
int GibbsSampling::choose_next(cv::Mat& p) {
	// setup random variables
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> randu(0.0, 1.0);

	std::vector<float> cdf;
	for (int i = 0; i < p.rows; ++i) {
		if (i == 0) {
			cdf.push_back(p.at<float>(i, 0));
		} else {
			cdf.push_back(cdf.back() + p.at<float>(i, 0));
		}
	}

	float r = randu(mt) * cdf.back();
	for (int i = 0; i < cdf.size(); ++i) {
		if (r < cdf[i]) return i;
	}

	return cdf.size() - 1;
}

/**
 * Normalize matrix such that each column has a mean 0 and a variance 1.
 */
void GibbsSampling::normalize(cv::Mat& x) {
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

/**
 * Objective function E() defined in eq(4).
 */
float GibbsSampling::E(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q) {
	std::cout << w << std::endl;
	std::cout << wh << std::endl;

	float ret = cv::sum((w - wh).mul(w - wh))[0] * 0.5 / a / a;

	std::cout << ret << std::endl;
	std::cout << x << std::endl;

	ret += cv::sum(x.mul(x))[0] * 0.5 / theta / theta;
	for (int i = 0; i < x.rows; ++i) {
		for (int j = i+1; j < x.rows; ++j) {
			for (int k = 0; k < w.rows; ++k) {
				float hoge = q.at<float>(i, j, k);
				ret -= (1.0f - q.at<float>(i, j, k)) * w.row(k).dot(x.row(j) - x.row(i));
			}
		}
	}

	for (int i = 0; i < x.rows; ++i) {
		for (int j = i + 1; x.rows; ++j) {
			for (int k = 0; k < w.rows; ++k) {
				ret += logf(1.0f + expf(w.row(k).dot(x.row(j) - x.row(i))));
			}
		}
	}

	return ret;
}

cv::Mat GibbsSampling::dEx(cv::Mat& x, cv::Mat& w, cv::Mat& q) {
	cv::Mat ret = cv::Mat::zeros(x.rows, x.cols, CV_32F);
	for (int i = 0; i < ret.rows; ++i) {
		cv::Mat row_i = ret.colRange(0, ret.cols).rowRange(i, i+1);
		dExi(i, x, w, q, row_i);
	}

	return ret;
}

void GibbsSampling::dExi(int i, cv::Mat& x, cv::Mat& w, cv::Mat& q, cv::Mat& ret) {
	x.row(i).copyTo(ret);
	ret /= (theta * theta);
	for (int j = 0; j < x.rows; ++j) {
		if (j == i) continue;
		for (int k = 0; k < w.rows; ++k) {
			ret -= w.row(k) * q.at<float>(i, j, j);
		}
	}
	for (int j = i + 1; j < x.rows; ++j) {
		for (int k = 0; k < w.rows; ++k) {
			ret += w.row(k);
		}
	}
	for (int j = i + 1; j < x.rows; ++j) {
		for (int k = 0; k < w.rows; ++k) {
			float ex = expf(w.row(k).dot(x.row(j) - x.row(i)));
			ret -= w.row(k) * ex / (1.0f + ex);
		}
	}
	for (int j = 0; j < i; ++j) {
		for (int k = 0; k < w.rows; ++k) {
			float ex = expf(w.row(k).dot(x.row(i) - x.row(j)));
			ret += w.row(k) * ex / (1.0f + ex);
		}
	}
}

cv::Mat GibbsSampling::dEw(cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q) {
	cv::Mat ret = cv::Mat::zeros(w.rows, w.cols, CV_32F);
	for (int k = 0; k < ret.rows; ++k) {
		cv::Mat row_k = ret.colRange(0, ret.cols).rowRange(k, k+1);
		dEwk(k, x, w, wh, q, row_k);
	}

	return ret;
}

void GibbsSampling::dEwk(int k, cv::Mat& x, cv::Mat& w, cv::Mat& wh, cv::Mat& q, cv::Mat& ret) {
	((cv::Mat)((w.row(k) - wh.row(k)) / a / a)).copyTo(ret);
	for (int i = 0; i < x.rows; ++i) {
		for (int j = i + 1; j < x.rows; ++j) {
			ret -= (1 - q.at<float>(i, j, k)) * (x.row(j) - x.row(i));
		}
	}

	for (int i = 0; i < x.rows; ++i) {
		for (int j = i + 1; j < x.rows; ++j) {
			float ex = expf(w.row(k).dot(x.row(j) - x.row(i)));
			ret += (x.row(j) - x.row(i)) * ex / (1.0f + ex);
		}
	}
}
