#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "HCMCMC.h"

int main(int argc, char* argv[]) {
	if (argc < 7) {
		printf("Usage: %s <S> <N> <M> <T> <ws_0> <ws_1>\n\n", argv[0]);
		return 1;
	}

	srand(0);

	int N = atoi(argv[2]);
	int M = atoi(argv[3]);
	int S = atoi(argv[1]);
	int T = atoi(argv[4]);

	// set up the desired weight
	cv::Mat ws = cv::Mat::zeros(S, 1, CV_32F);
	ws.at<float>(0, 0) = atof(argv[5]);
	if (S > 1) {
		ws.at<float>(1, 0) = atof(argv[6]);
	}

	//for (int T = 100; T <= 1000; T += 100) {
		std::cout << "T: " << T << "..." << std::endl;

		clock_t start = clock();

		HCMCMC hc(N, M, S, 2, T, ws);
		hc.run();

		clock_t end = clock();
		std::cout << "Elapsed: " << (double)(end-start)/CLOCKS_PER_SEC << " [sec]" << std::endl;
		std::cout << std::endl;
	//}

	
	return 0;
}