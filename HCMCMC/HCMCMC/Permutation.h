#pragma once

#include <vector>

class Permutation {
private:
	int n;
	int m;

public:
	std::vector<int> data;

public:
	Permutation(int n, int m) : n(n), m(m) {
		for (int i = 0; i < n; ++i) {
			data.push_back(0);
		}
	}

public:
	bool next() {
		if (data[0] < m) {
			data[0]++;
			return true;
		} else {
			data[0] = 0;
			return carry_over(1);
		}
	}

private:
	bool carry_over(int d) {
		if (d >= data.size()) return false;

		if (data[d] < m) {
			data[d]++;
			return true;
		} else {
			data[d] = 0;
			return carry_over(d + 1);
		}
	}
};

