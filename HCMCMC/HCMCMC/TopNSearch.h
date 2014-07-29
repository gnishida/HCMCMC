#pragma once

#include <vector>
#include <queue>

class TopNSearch {
public:
	static enum { ORDER_ASC = 0, ORDER_DESC };

private:
	std::priority_queue<std::pair<float, int>> data;
	//std::vector<float> data;

public:
	TopNSearch() {}
	~TopNSearch() {}

	void add(int key, float value) {
		data.push(std::pair<float, int>(value, key));
		//data.push_back(value);
	}

	std::vector<int> topN(int n, int orderType) {
		std::vector<int> ret;
		/*std::priority_queue<std::pair<float, int>> q;
		for (int i = 0; i < data.size(); ++i) {
			q.push(std::pair<float, int>(data[i], i));
		}*/

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

