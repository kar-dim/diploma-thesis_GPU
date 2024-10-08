﻿#include "Utilities.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

using namespace cimg_library;
using std::string;

string Utilities::loadFileString(const string& input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open())
		throw std::runtime_error("Could not load Program: " + input + "\n");
	return string(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
}

string Utilities::addSuffixBeforeExtension(const string& file, const string& suffix) 
{
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

namespace timer 
{
	void start() 
	{
		startTime = std::chrono::high_resolution_clock::now();
	}
	void end() 
	{
		currentTime = std::chrono::high_resolution_clock::now();
	}
	float elapsedSeconds() 
	{
		return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() / 1000000.0f);
	}
}

//see https://blog.bearcats.nl/accurate-sleep-function/
void Utilities::accurateSleep(double seconds) 
{
	double estimate = 5e-3, mean = 5e-3, m2 = 0;
	long long count = 1;
	while (seconds > estimate) 
	{
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