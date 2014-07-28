#include "HC.h"

/**
 * 行列xtは、N行S列の行列で、真の値。各行が各データポイントを表す。
 * 各列は、各コンポーネントの値を表す。
 *
 * 行列zpは、N行D列の行列で、各行がデータポイントを表す。
 * 各列は、各次元の座標を表す。
 *
 * 行列whは、M行S列の行列で、ユーザ毎の重みベクトルを表す。
 */
cv::Mat HC::run(cv::Mat& xt, cv::Mat& zp, cv::Mat& wh) {
	int sizes[3];
	sizes[0] = xt.rows;
	sizes[1] = xt.rows;
	sizes[2] = wh.rows;

	cv::Mat q = cv::Mat(3, sizes, CV_32F);
	q *= 0.0f;

	for (int i = 0; i < zp.rows; ++i) {
		for (int j = i + 1; j < zp.rows; ++j) {
			for (int k = 0; k < wh.rows; ++k) {
				if (wh.row(k).dot(xt.row(i) - xt.row(j)) > 0) {
					q.at<float>(cv::Vec3i(i, j, k)) = 1.0f;
				}
				q.at<float>(cv::Vec3i(j, i, k)) = 1.0f - q.at<float>(cv::Vec3i(i, j, k));
			}
		}
	}

	return q;
}
