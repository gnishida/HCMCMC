#include "HCMCMC.h"
#include <vector>
#include <random>
#include "GradientDescent.h"
#include "Util.h"
#include "KSTest.h"
#include "HC.h"
#include "TopNSearch.h"

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
	z[0] = 2;
	z[1] = 2;

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


			// 真の値を取得
			cv::Mat xt = getTrueValue(zp);
			
			// synthesize q
			cv::Mat q = HC::run(img, zp, wh, xt, true);

			// initialize x, w
			cv::Mat x = cv::Mat::zeros(N, S, CV_32F);
			cv::Mat w(wh.rows, wh.cols, CV_32F);
			wh.copyTo(w);

			GradientDescent gd;
			gd.run(x, w, wh, q, 100);

			//std::cout << x << std::endl;

			// compute the accuracy of the estimation
			//check_estimation(x, xt);

			// choose the next state according to the conditional distribution
			cv::Mat score = x * ws;
			z[d] = zp.at<float>(choose_next(score), d);
		}
	}

	// Kolmogorov-Smirnov test
	//std::cout << "K-S test: " << KStest(result) << std::endl;

	// top 10%
	std::cout << "Top 10%: " << top10(result) * 100.0f << " %" << std::endl;

	// save the result
	char filename[256];
	sprintf(filename, "result_%d.jpg", T);
	save(result, filename);
}

/**
 * 与えられたサンプルデータポイントについて、
 * 真の値を取得して返却する。
 */
cv::Mat HCMCMC::getTrueValue(cv::Mat& zp) {
	cv::Mat xt = cv::Mat::zeros(zp.rows, img.size(), CV_32F);
	for (int i = 0; i < xt.rows; ++i) {
		for (int s = 0; s < xt.cols; ++s) {
			xt.at<float>(i, s) = img[s].at<uchar>((int)zp.at<float>(i, 1), (int)zp.at<float>(i, 0));
		}
	}
	Util::normalize(xt);

	return xt;
}

/**
 * 条件分布pに従い、次の状態を決定する。
 */
int HCMCMC::choose_next(cv::Mat& p) {
	//std::cout << p << std::endl;

	// adjust the score scale to start from 0
	double minVal, maxVal;
	cv::minMaxLoc(p, &minVal, &maxVal);
	p -= minVal;

	// おまじない。これがないと、一番下の確率のやつは、確率0となっちゃう
	p += 0.22f;

	//std::cout << p << std::endl;

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
	std::cout << result << std::endl;

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
	std::cout << "K-S test (X): " << test1 << std::endl;

	//////////////////////////////////////////////////////////////////
	// Y -> X order

	// create F()
	F_total = 0.0f;
	for (int c = 0; c < result.cols; ++c) {
		for (int r = 0; r < result.rows; ++r) {
			float expected = 0.0f;
			for (int s = 0; s < S; ++s) {
				expected += ws.at<float>(s, 0) * img[s].at<uchar>(r, c) / total_img[s];
			}
			F_total += expected;
			F[c * result.rows + r] = F_total;
		}
	}

	// create Fn()
	Fn_total = 0.0f;
	for (int c = 0; c < result.cols; ++c) {
		for (int r = 0; r < result.rows; ++r) {
			Fn_total += result.at<float>(r, c);
			Fn[c * result.rows + r] = Fn_total;
		}
	}

	// normalize Fn()
	for (int c = 0; c < result.cols; ++c) {
		for (int r = 0; r < result.rows; ++r) {
			Fn[c * result.rows + r] /= Fn_total;
		}
	}
	std::cout << "Fn_total: " << Fn_total << std::endl;

	float test2 = KSTest::test(Fn, F, Fn_total);
	std::cout << "K-S test (Y): " << test2 << std::endl;


	// take the largest
	return std::max(test1, test2);
}

/**
 * Top10%のデータポイントのうち、どのぐらいをサンプリングできたか？
 */
float HCMCMC::top10(cv::Mat& result) {
	std::cout << result << std::endl;
	TopNSearch tns;
	for (int r = 0; r < img[0].rows; ++r) {
		for (int c = 0; c < img[0].cols; ++c) {
			float value = 0.0f;
			for (int s = 0; s < S; ++s) {
				value += (float)img[s].at<uchar>(r, c) * ws.at<float>(s, 0);
			}

			tns.add(r * img[0].cols + c, value);
		}
	}

	std::vector<int> top = tns.topN(img[0].rows * img[0].cols * 0.1, TopNSearch::ORDER_DESC);

	int cnt = 0;
	for (int i = 0; i < top.size(); ++i) {
		int r = top[i] / img[0].cols;
		int c = top[i] % img[0].cols;

		if (result.at<float>(r, c) > 0.0f) {
			cnt++;
		}
	}

	return (float)cnt / (float)top.size();
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
