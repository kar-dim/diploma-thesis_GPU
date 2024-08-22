#pragma once
#include "cimg_init.h"
#include <arrayfire.h>
#include <chrono>
#include <string>

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class Utilities {
public:
	static std::string add_suffix_before_extension(const std::string& file, const std::string& suffix);
	static void accurate_timer_sleep(double seconds);
	template<typename T>
	static af::array cimg_yuv_to_afarray(const cimg_library::CImg<T>& cimg_image) {
		return af::transpose(af::array(cimg_image.width(), cimg_image.height(), cimg_image.get_channel(0).data()).as(af::dtype::f32));
	}
};

/*!
 *  \brief  simple methods to calculate execution times
 *  \author Dimitris Karatzas
 */
namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	float secs_passed();
}