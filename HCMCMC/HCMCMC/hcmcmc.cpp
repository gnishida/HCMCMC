#include "HCMCMC.h"
#include <vector>
#include <random>
#include "GradientDescent.h"
#include "Util.h"

HCMCMC::HCMCMC(int N, int M, int S, int D, int T, cv::Mat& ws) : N(N), M(M), S(S), D(D), T(T), ws(ws) {
}

void HCMCMC::run() {
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

	// initialize the result
	cv::Mat result = cv::Mat(img[0].rows, img[0].cols, CV_32F, cv::Scalar(0.0f));

	// initialize the state (center)
	std::vector<int> z;
	for (int d = 0; d < D; ++d) {
		z.push_back(img[0].rows * 0.5f);
	}

	// synthesize wh
	cv::Mat wh(M, S, CV_32F);
	cv::randu(wh, cv::Scalar(0.0f), cv::Scalar(1.0f));

	// normalize wh
	/*
	cv::Mat wh_col;
	cv::reduce(wh, wh_col, 1, CV_REDUCE_SUM);
	for (int k = 0; k < M; ++k) {
		for (int s = 0; s < S; ++s) {
			wh.at<float>(k, s) /= wh_col.at<float>(k, 0);
		}
	}
	*/

	// make the first user's weight as uniform
	for (int s = 0; s < S; ++s) {
		wh.at<float>(0, s) = 1.0f / S;
	}

	// MCMC
	for (int t = 0; t < T; ++t) {
		for (int d = 0; d < D; ++d) {
			// record the current state
			result.at<float>(z[0], z[1]) += 1.0f;
			//std::cout << "sampled: " << z[0] << "," << z[1] << std::endl;

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
			cv::Mat est_x = grad_desc_test(wh, zp);

			// choose the next state according to the conditional distribution
			cv::Mat est_score = est_x * ws;
			z[d] = zp.at<float>(choose_next(est_score), d);
		}
	}

	// save the result
	char filename[256];
	sprintf(filename, "result_%d.jpg", T);
	save(result, filename);

	// Kolmogorov-Smirnov test
	std::cout << "K-S test: " << KStest(result) << std::endl;
}

/**
 * Gradient descent
 */
cv::Mat HCMCMC::grad_desc_test(cv::Mat& wh, cv::Mat& zp) {
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
	Util::normalize(xt);

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
				if (wh.row(k).dot(xt.row(i) - xt.row(j)) > 0) {
					q.at<float>(cv::Vec3i(i, j, k)) = 1.0f;
				}
				q.at<float>(cv::Vec3i(j, i, k)) = 1.0f - q.at<float>(cv::Vec3i(i, j, k));
			}
		}
	}

	// initialize x, w
	cv::Mat x = cv::Mat::zeros(xt.rows, xt.cols, CV_32F);
	cv::Mat w(wh.rows, wh.cols, CV_32F);
	wh.copyTo(w);

	GradientDescent gd;
	gd.run(x, w, wh, q, 100);

	// compute the accuracy of the estimation
	check_estimation(x, wh, q);

	return x;
}

/**
 * Choose the next state
 */
int HCMCMC::choose_next(cv::Mat& p) {
	// setup random variables
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> randu(0.0, 1.0);

	// adjust the score scale to start from 0
	double minVal, maxVal;
	cv::minMaxLoc(p, &minVal, &maxVal);
	p -= minVal;

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

void HCMCMC::save(cv::Mat& result, char* filename) {
	double minVal, maxVal;
	cv::minMaxLoc(result, &minVal, &maxVal);
	result *= 255.0f / maxVal * 1.2f;
	//float avg = cv::mean(result)[0];
	//result *= 127.0f / avg;

	cv::Mat resultImg;
	result.convertTo(resultImg, CV_8U);

	cv::imwrite(filename, resultImg);
}

/**
 * Kolmogorov-Smirnov test
 */
float HCMCMC::KStest(cv::Mat& result) {
	std::vector<float> total_img;
	for (int s = 0; s < S; ++s) {
		total_img.push_back(cv::sum(img[s])[0]);
	}

	// create F()
	std::vector<float> F(result.rows * result.cols);
	float F_total = 0.0f;
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			float expected = 0.0f;
			for (int s = 0; s < S; ++s) {
				expected += ws.at<float>(s, 0) * img[s].at<uchar>(r, c) / total_img[s];
			}
			F_total += expected;
			F[r * result.cols + c] = F_total;
		}
	}

	// normalize F()
	for (int i = 0; i < F.size(); ++i) {
		F[i] /= F_total;
	}

	// create Fn()
	std::vector<float> Fn(result.rows * result.cols);
	float Fn_total = 0.0f;
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			Fn_total += result.at<float>(r, c) / (float)T;
			Fn[r * result.cols + c] = Fn_total;
		}
	}

	// compute D
	float D = 0.0f;
	for (int i = 0; i < F.size(); ++i) {
		if (fabs(Fn[i] - F[i]) > D) {
			D = fabs(Fn[i] - F[i]);
		}
	}

	return D * sqrtf((float)T);
}

void HCMCMC::check_estimation(cv::Mat& est, cv::Mat& wh, cv::Mat& q) {
	int correct = 0;
	int incorrect = 0;

	for (int i = 0; i < N; ++i) {
		for (int j = i + 1; j < N; ++j) {
			for (int k = 0; k < M; ++k) {
				if (wh.row(k).dot(est.row(i) - est.row(j)) > 0) {
					if (q.at<float>(i, j, k) == 1.0) {
						correct++;
					} else {
						incorrect++;
					}
				} else {
					if (q.at<float>(i, j, k) == 1.0) {
						incorrect++;
					} else {
						correct++;
					}
				}
			}
		}
	}
	
	float ratio = (float)correct / (float)(correct + incorrect);
	if (ratio < 0.8f) {
		std::cout << "[Warning] Correct ratio: " << ratio*100.0f << " [%]" << std::endl;
	}
}
