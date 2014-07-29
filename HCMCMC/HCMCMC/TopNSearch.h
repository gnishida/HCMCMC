#pragma once

#include <vector>
#include <queue>

template<class T>
class TopNSearch {
public:
	static enum { ORDER_ASC = 0, ORDER_DESC };

private:
	std::priority_queue<std::pair<float, T>> data;

public:
	TopNSearch() {}
	~TopNSearch() {}

	void add(T key, float value) {
		data.push(std::pair<float, T>(value, key));
	}

	std::vector<T> topN(int n, int orderType = ORDER_DESC) {
		std::vector<T> ret;

		for (int i = 0; i < n; ++i) {
			ret.push_back(data.top().second);
			data.pop();
		}

		return ret;
	}

	size_t size() const {
		return data.size();
	}
};

