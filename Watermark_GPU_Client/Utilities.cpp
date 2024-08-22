#include "Utilities.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <thread>

using namespace cimg_library;
using std::string;

namespace timer {
	void start() {
		start_timex = std::chrono::high_resolution_clock::now();
	}
	void end() {
		cur_timex = std::chrono::high_resolution_clock::now();
	}
	float secs_passed() {
		return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(cur_timex - start_timex).count() / 1000000.0f);
	}
}

string Utilities::add_suffix_before_extension(const string& file, const string& suffix) {
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

//see https://blog.bearcats.nl/accurate-sleep-function/
void Utilities::accurate_timer_sleep(double seconds) {
	double estimate = 5e-3, mean = 5e-3, m2 = 0;
	long long count = 1;
	while (seconds > estimate) {
		auto start = std::chrono::high_resolution_clock::now();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		auto end = std::chrono::high_resolution_clock::now();
		double observed = (end - start).count() / 1e9;
		seconds -= observed;
		++count;
		double delta = observed - mean;
		mean += delta / count;
		m2 += delta * (observed - mean);
		double stddev = sqrt(m2 / (count - 1));
		estimate = mean + stddev;
	}
	// spin lock
	auto start = std::chrono::high_resolution_clock::now();
	while ((std::chrono::high_resolution_clock::now() - start).count() / 1e9 < seconds);
}