﻿#pragma once
#include "opencl_init.h"
#include <af/opencl.h>
#include <string>
#include <utility>

enum MASK_TYPE {
	ME,
	NVF
};

enum IMAGE_TYPE {
	RGB,
	GRAYSCALE
};

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class Watermark {

private:
	static constexpr int Rx_mappings[64]{
		0,  1,  2,  3,  4,  5,  6,  7,
		1,  8,  9,  10, 11, 12, 13, 14,
		2,  9,  15, 16, 17, 18, 19, 20,
		3,  10, 16, 21, 22, 23, 24, 25,
		4,  11, 17, 22, 26, 27, 28, 29,
		5,  12, 18, 23, 27, 30, 31, 32,
		6,  13, 19, 24, 28, 31, 33, 34,
		7,  14, 20, 25, 29, 32, 34, 35
	};
	const cl::Context context{ afcl::getContext(true) };
	const cl::CommandQueue queue{ afcl::getQueue(true) };
	const cl::Program program_me, program_custom;
	const std::string w_file_path, custom_kernel_name;
	const int p, p_squared, p_squared_minus_one, p_squared_minus_one_squared, pad;
	const float psnr;
	af::array rgb_image, image, w;
	dim_t rows, cols;

	af::array calculate_neighbors_array(const af::array& array, const int p, const int p_squared, const int pad) const;
	std::pair<af::array, af::array> correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int padded_cols) const;
	float calculate_correlation(const af::array& e_u, const af::array& e_z) const;
	af::array compute_custom_mask(const af::array &image) const;
	af::array compute_prediction_error_mask(const af::array& image, af::array& error_sequence, af::array& coefficients, const bool mask_needed) const;
	af::array compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& error_sequence) const;
	af::array calculate_error_sequence(const af::array& u, const af::array& coefficients) const;
	cl::Image2D copyBufferToImage(const cl_mem* image_buff, const dim_t rows, const dim_t cols) const;
public:
	Watermark(const af::array& rgb_image, const af::array &image, const std::string &w_file_path, const int p, const float psnr, const cl::Program &program_me, const cl::Program &program_custom, const std::string &custom_kernel_name);
	Watermark(const std::string &w_file_path, const int p, const float psnr, const cl::Program& program_me, const cl::Program& program_custom, const std::string custom_kernel_name);
	af::array load_W(const dim_t rows, const dim_t cols) const;
	void load_image(const af::array& image);
	af::array make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE mask_type, IMAGE_TYPE image_type) const;
	float mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type) const;
	float mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients) const;
	static void display_array(const af::array& array, const int width = 1600, const int height = 900);
};