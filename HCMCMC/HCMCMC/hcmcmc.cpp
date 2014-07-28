#include "HCMCMC.h"
#include <vector>
#include <random>
#include "GradientDescent.h"
#include "Util.h"
#include "KSTest.h"
#include "HC.h"

HCMCMC::HCMCMC(int N, int M, int S, int D, int T, cv::Mat& ws) : N(N), M(M), S(S), D(D), T(T), ws(ws) {
}

void HCMCMC::run() {
	// setup images (which will be used as probability distribution)
	img.clear();
	for (int s = 0; s < S; ++s) {
		char filename[256];
		sprintf(filename, "truth%d.bmp", s);
		img.push_back(cv::imread(filename, 0));
	}

	// initialize the result
	cv::Mat result = cv::Mat(img[0].cols, img[0].rows, CV_32F, cv::Scalar(0.0f));

	// initialize the state (center)
	std::vector<int> z;
	for (int d = 0; d < D; ++d) {
		z.push_back(img[0].rows * 0.5f);
	}

	// synthesize wh
	cv::Mat wh(M, S, CV_32F);
	for (int k = 0; k < M; ++k) {
		for (int s = 0; s < S; ++s) {
			wh.at<float>(k, s) = Util::randu(0.0f, 1.0f);
		}
	}

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
	/*for (int s = 0; s < S; ++s) {
		wh.at<float>(0, s) = 1.0f / S;
	}*/

	/*
	wh.at<float>(0, 0) = 1.0f;
	wh.at<float>(0, 1) = 0.0f;
	wh.at<float>(1, 0) = 0.0f;
	wh.at<float>(1, 1) = 1.0f;
	*/

	//std::cout << wh << std::endl;


	// MCMC
	for (int t = 0; t < T; ++t) {
		for (int d = 0; d < D; ++d) {
			// record the current state
			result.at<float>(z[1], z[0]) += 1.0f;
			//std::cout << "sampled: " << z[0] << "," << z[1] << std::endl;

			// sample N data
			cv::Mat zp(N, D, CV_32F);
			for (int i = 0; i < N; ++i) {
				for (int d2 = 0; d2 < D; ++d2) {
					if (d2 == d) {
						zp.at<float>(i, d2) = i;//(int)(randu(mt) * img[0].rows);
					} else {
						zp.at<float>(i, d2) = z[d2];
					}
				}
			}



			//std::cout << zp << std::endl;



			// find the optimum by gradient descent
			cv::Mat est_x = grad_desc_test(wh, zp);

			// choose the next state according to the conditional distribution
			cv::Mat est_score = est_x * ws;
			z[d] = zp.at<float>(choose_next(est_score), d);
		}
	}

	// Kolmogorov-Smirnov test
	std::cout << "K-S test: " << KStest(result) << std::endl;

	// save the result
	char filename[256];
	sprintf(filename, "result_%d.jpg", T);
	save(result, filename);
}

/**
 * Gradient descent
 */
cv::Mat HCMCMC::grad_desc_test(cv::Mat& wh, cv::Mat& zp) {
	// given the N samples, compute the true scores
	cv::Mat xt = cv::Mat::zeros(N, S, CV_32F);
	for (int i = 0; i < N; ++i) {
		for (int s = 0; s < S; ++s) {
			//xt.at<float>(i, s) = img[s].at<uchar>((int)zp.at<float>(i, 0), (int)zp.at<float>(i, 1));
			xt.at<float>(i, s) = img[s].at<uchar>((int)zp.at<float>(i, 1), (int)zp.at<float>(i, 0));
		}
	}
	Util::normalize(xt);
	//std::cout << xt << std::endl;

	// synthesize q
	int sizes[3];
	sizes[0] = N;
	sizes[1] = N;
	sizes[2] = M;
	cv::Mat q = HC::run(xt, zp, wh);

	/*
	// ユーザ投票結果を捏造する
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			std::cout << q.at<float>(i, j, 0) << ",";
		}
		std::cout << std::endl;
	}
	*/


	// initialize x, w
	cv::Mat x = cv::Mat::zeros(xt.rows, xt.cols, CV_32F);
	cv::Mat w(wh.rows, wh.cols, CV_32F);
	wh.copyTo(w);

	GradientDescent gd;
	gd.run(x, w, wh, q, 100, xt);

	//std::cout << x << std::endl;

	// compute the accuracy of the estimation
	check_estimation(x, xt);

	return x;
}

/**
 * Choose the next state
 */
int HCMCMC::choose_next(cv::Mat& p) {
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

	float r = Util::randu(0.0f, cdf.back());
	for (int i = 0; i < cdf.size(); ++i) {
		if (r < cdf[i]) {
			return i;
		}
	}

	return cdf.size() - 1;
}

void HCMCMC::save(cv::Mat result, char* filename) {
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

	//////////////////////////////////////////////////////////////////
	// X -> Y order

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

	// create Fn()
	std::vector<float> Fn(result.rows * result.cols);
	float Fn_total = 0.0f;
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			Fn_total += result.at<float>(r, c);
			Fn[r * result.cols + c] = Fn_total;
		}
	}

	// normalize Fn()
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			Fn[r * result.cols + c] /= Fn_total;
		}
	}

	float test1 = KSTest::test(Fn, F, Fn_total);

	//////////////////////////////////////////////////////////////////
	// Y -> X order

	// create F()
	F_total = 0.0f;
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

	// create Fn()
	Fn_total = 0.0f;
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			Fn_total += result.at<float>(r, c);
			Fn[r * result.cols + c] = Fn_total;
		}
	}

	// normalize Fn()
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			Fn[r * result.cols + c] /= Fn_total;
		}
	}
	std::cout << "Fn_total: " << Fn_total << std::endl;

	float test2 = KSTest::test(Fn, F, Fn_total);

	// take the largest
	return std::max(test1, test2);
}

void HCMCMC::check_estimation(cv::Mat& x, cv::Mat& xt) {
	int correct = 0;
	int incorrect = 0;

	for (int i = 0; i < N; ++i) {
		for (int j = i + 1; j < N; ++j) {
			for (int s = 0; s < S; ++s) {
				//if (s > 0) break;
				if (xt.at<float>(i, s) > xt.at<float>(j, s)) {
					if (x.at<float>(i, s) > x.at<float>(j, s)) {
						correct++;
					} else {
						incorrect++;
					}
				} else {
					if (x.at<float>(i, s) > x.at<float>(j, s)) {
						incorrect++;
					} else {
						correct++;
					}
				}
			}
		}
	}
	
	float ratio = (float)correct / (float)(correct + incorrect);
	if (ratio < 0.95f) {
		//std::cout << "[Warning] Correct ratio: " << ratio*100.0f << " [%]" << std::endl;
	}
}
