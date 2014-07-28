#include "KSTest.h"

/**
 * Kolmogorov-Smirnov test
 * 
 * @param Fn	実験結果の分布 (0から1に正規化されていること)
 * @param F		元の分布 (0から1に正規化されていること)
 * @param N		サンプル数
 */
float KSTest::test(std::vector<float>& Fn, std::vector<float>& F, int N) {
	// compute D
	float D = 0.0f;
	for (int i = 0; i < F.size(); ++i) {
		if (fabs(Fn[i] - F[i]) > D) {
			D = fabs(Fn[i] - F[i]);
		}
	}

	return D * sqrtf((float)N);
}
