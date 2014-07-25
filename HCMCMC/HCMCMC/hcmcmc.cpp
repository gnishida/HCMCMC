#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "GibbsSampling.h"

int main(int argc, char* argv[]) {
	if (argc < 4) {
		printf("Usage: %s <S> <N> <M>\n\n", argv[0]);
		return 1;
	}

	int N = atoi(argv[2]);
	int M = atoi(argv[3]);
	int S = atoi(argv[1]);

	// set up the desired weight
	cv::Mat ws = cv::Mat::zeros(S, 1, CV_32F);
	ws.at<float>(1, 0) = 1.0f;

	for (int T = 100; T <= 1000; T += 100) {
		clock_t start = clock();

		GibbsSampling gs(N, M, S, 2, T, ws);
		gs.run();

		clock_t end = clock();
		printf("Elapsed: %f [sec]\n", (double)(end-start)/CLOCKS_PER_SEC);
		printf("\n");
	}

	
	return 0;
}