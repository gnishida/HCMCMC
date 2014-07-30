#include "HCMCMC.h"
#include <vector>
#include <random>
#include "GradientDescent.h"
#include "Util.h"
#include "KSTest.h"
#include "HC.h"
#include "TopNSearch.h"
#include "Permutation.h"

HCMCMC::HCMCMC(int N, int M, int S, int D, int T, cv::Mat& ws) : N(N), M(M), S(S), D(D), T(T), ws(ws) {
}

void HCMCMC::run() {
	std::vector<int> size(D);
	for (int i = 0; i < D; ++i) {
		size[i] = N;
	}

	// setup the ground truth
	setupGroundTruth(size);

	// initialize the result
	cv::Mat result = cv::Mat(D, size.data(), CV_32F, cv::Scalar(0.0f));

	// initialize the state (center)
	std::vector<int> z;
	for (int d = 0; d < D; ++d) {
		z.push_back(N * 0.5f);
	}

	// synthesize wh
	cv::Mat wh(M, S, CV_32F);
	for (int k = 0; k < M; ++k) {
		for (int s = 0; s < S; ++s) {
			wh.at<float>(k, s) = Util::randu();
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

			//std::cout << xt << std::endl;
			
			// synthesize q
			cv::Mat q = HC::run(zp, wh, xt, false);

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
			cv::Mat score = xt * ws;

			z[d] = zp.at<float>(choose_best(score), d);

			//std::cout << "best: "; Util::displayVector(z);

			//Util::displayVector(z);

			// record the local best
			result.at<float>(z.data()) += 1.0f;

			//std::cout << score << std::endl;
			z[d] = zp.at<float>(choose_next(score), d);

			// record the current state
			//result.at<float>(z.data()) += 1.0f;
		}
	}

	// Kolmogorov-Smirnov test
	//std::cout << "K-S test: " << KStest(result) << std::endl;

	// compute the expected
	cv::Mat truth = cv::Mat(D, size.data(), CV_32F, cv::Scalar(0.0f));
	for (int s = 0; s < S; ++s) {
		truth += img[s] * ws.at<float>(s, 0);
	}

	//Util::displayMat3f(truth, N);

	// top 5%
	std::cout << "Top 5%: " << topN(result, truth, 0.05f) * 100.0f << " %" << std::endl;

	// save the result
	char filename[256];
	sprintf(filename, "result_%d.jpg", T);
	save(result, 0, 1, filename);
	
	// save the ground truth
	sprintf(filename, "truth_%d.jpg", T);
	save(truth, 0, 1, filename);
}

/**
 * 真のスコア分布を作成する。
 */
void HCMCMC::setupGroundTruth(std::vector<int>& size) {
	for (int s = 0; s < S; ++s) {
		img.push_back(cv::Mat(D, size.data(), CV_32F, cv::Scalar(0.0f)));
	}

	Permutation perm(D, N-1);
	while (true) {
		std::vector<int> point(D);
		for (int i = 0; i < D; ++i) {
			point[i] = perm.data[i];
		}
		for (int s = 0; s < S; ++s) {
			img[s].at<float>(point.data()) = Util::randu();
		}

		if (!perm.next()) break;
	}

	// normalize
	for (int s = 0; s < S; ++s) {
		img[s] /= cv::sum(img[s])[0];
		//std::cout << img[s] << std::endl;
	}

	// debug
	/*for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				std::cout << i << "," << j << "," << k << " [";
				for (int s = 0; s < S; ++s) {
					std::cout << img[s].at<float>(i, j, k) << ",";
				}
				std::cout << "]" << std::endl;
			}
		}
	}*/
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
	//p += 0.22f;

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

/**
 * 条件分布pにおいて、ベストの状態を探す。
 */
int HCMCMC::choose_best(cv::Mat& p) {
	float best = -std::numeric_limits<float>::max();
	int best_index = -1;

	for (int i = 0; i < p.rows; ++i) {
		if (p.at<float>(i, 0) > best) {
			best = p.at<float>(i, 0);
			best_index = i;
		}
	}

	return best_index;
}

/**
 * 与えられたサンプルデータポイントについて、
 * 真の値を取得して返却する。
 */
cv::Mat HCMCMC::getTrueValue(cv::Mat& zp) {
	cv::Mat xt = cv::Mat::zeros(N, S, CV_32F);
	for (int i = 0; i < N; ++i) {
		for (int s = 0; s < S; ++s) {
			std::vector<int> point(D);
			for (int d = 0; d < D; ++d) {
				point[d] = zp.at<float>(i, d);
			}
			xt.at<float>(i, s) = img[s].at<float>(point.data());//img[s].at<uchar>((int)zp.at<float>(i, 1), (int)zp.at<float>(i, 0));
		}
	}
	//Util::normalize(xt);

	return xt;
}

/**
 * 結果の入った行列resutlのうち、d1次元とd2次元を画像として保存する。
 */
void HCMCMC::save(cv::Mat result, int d1, int d2, char* filename) {
	// 指定されたプレーンでの断面を２次元の行列に格納する
	// (d1次元、d2次元以外は、真ん中の座標を使用する)
	cv::Mat plane(N, N, CV_32F);
	std::vector<int> point(D);
	for (int d = 0; d < D; ++d) {
		point[d] = (int)(N * 0.5f);
	}
	for (int r = 0; r < N; ++r) {
		for (int c = 0; c < N; ++c) {
			point[d1] = r;
			point[d2] = c;
			plane.at<float>(r, c) = result.at<float>(point.data());
		}
	}

	// ２次元の行列の要素の値が0～255の範囲になるよう調節する
	double minVal, maxVal;
	cv::minMaxLoc(plane, &minVal, &maxVal);
	plane *= 255.0f / maxVal * 1.2f;

	// uchar型の行列に変換する
	cv::Mat resultImg;
	plane.convertTo(resultImg, CV_8U);

	cv::imwrite(filename, resultImg);
}

/**
 * Kolmogorov-Smirnov test
 */
/*
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
*/

/**
 * Top10%のデータポイントのうち、どのぐらいをサンプリングできたか？
 */
float HCMCMC::topN(cv::Mat& result, cv::Mat& truth, float topRatio) {
	//std::cout << result << std::endl;

	TopNSearch<std::vector<int> > tns;
	Permutation perm(D, N - 1);


	while (true) {
		std::vector<int> point(D);
		for (int d = 0; d < D; ++d) {
			point[d] = perm.data[d];
		}
		tns.add(perm.data, truth.at<float>(point.data()));

		if (!perm.next()) break;
	}

	std::vector<std::vector<int> > top = tns.topN(ceilf(powf(N, D) * topRatio));

	int cnt = 0;
	for (int i = 0; i < top.size(); ++i) {
		if (result.at<float>(top[i].data()) > 0.0f) {
			cnt++;
		}
	}

	return (float)cnt / (float)top.size();
}

/*void HCMCMC::check_estimation(cv::Mat& x, cv::Mat& xt) {
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
*/