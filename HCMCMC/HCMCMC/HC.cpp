﻿#include "HC.h"
#include "Util.h"

/**
 * 行列xtは、N行S列の行列で、真の値。各行が各データポイントを表す。
 * 各列は、各コンポーネントの値を表す。
 *
 * 行列zpは、N行D列の行列で、各行がデータポイントを表す。
 * 各列は、各次元の座標を表す。
 *
 * 行列whは、M行S列の行列で、ユーザ毎の重みベクトルを表す。
 */
cv::Mat HC::run(cv::Mat& zp, cv::Mat& wh, cv::Mat& xt, bool noise) {
	float rel = 10.0f;

	int sizes[3];
	sizes[0] = xt.rows;
	sizes[1] = xt.rows;
	sizes[2] = wh.rows;

	// ユーザからの回答行列Qを計算する
	cv::Mat q = cv::Mat(3, sizes, CV_32F);
	q *= 0.0f;

	for (int i = 0; i < zp.rows; ++i) {
		for (int j = i + 1; j < zp.rows; ++j) {
			for (int k = 0; k < wh.rows; ++k) {
				if (noise) {
					if (1.0f / (1.0f + expf(rel * wh.row(k).dot(xt.row(j) - xt.row(i)))) >= Util::randu(0.0f, 1.0f)) {
						q.at<float>(cv::Vec3i(i, j, k)) = 1.0f;
					}
				} else {
					if (wh.row(k).dot(xt.row(i) - xt.row(j)) > 0) {
						q.at<float>(cv::Vec3i(i, j, k)) = 1.0f;
					}
				}
				q.at<float>(cv::Vec3i(j, i, k)) = 1.0f - q.at<float>(cv::Vec3i(i, j, k));
			}
		}
	}

	return q;
}
