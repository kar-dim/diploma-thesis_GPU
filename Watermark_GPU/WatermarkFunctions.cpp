﻿#include "WatermarkFunctions.h"
#include <fstream>
#include <arrayfire.h>
#include <iostream>
#include <string>
#include <af/opencl.h>
#include <cmath>
#include <memory>
#include <functional>

using std::cout;

//constructor without specifying input image yet, it must be supplied later by calling the appropriate public method
WatermarkFunctions::WatermarkFunctions(const std::string w_file_path, const int p, const float psnr, const cl::Program& prog_me, const cl::Program& prog_custom, const std::string custom_kernel_name)
		:p(p), p_squared(p*p), p_squared_minus_one(p_squared-1), p_squared_minus_one_squared(p_squared_minus_one * p_squared_minus_one), pad(p/2), psnr(psnr), w_file_path(w_file_path), custom_kernel_name(custom_kernel_name), program_me(prog_me), program_custom(prog_custom) {
	this->rows = -1;
	this->cols = -1;
}

//full constructor
WatermarkFunctions::WatermarkFunctions(const af::array& image, const std::string w_file_path, const int p, const float psnr, const cl::Program& program_me, const cl::Program& program_custom, const std::string custom_kernel_name)
		:WatermarkFunctions::WatermarkFunctions(w_file_path, p, psnr, program_me, program_custom, custom_kernel_name) {
	load_image(image);
	load_W(this->rows, this->cols);
}

//supply the input image to apply watermarking and detection
void WatermarkFunctions::load_image(const af::array& image) {
	this->image = image;
	this->rows = image.dims(0);
	this->cols = image.dims(1);
}

//helper method to load the random noise matrix W from the file specified.
void WatermarkFunctions::load_W(const dim_t rows, const dim_t cols) {
	std::ifstream w_stream(this->w_file_path.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		std::string error_str("Error opening '" + this->w_file_path + "' file for Random noise W array");
		throw std::exception(error_str.c_str());
	}
	w_stream.seekg(0, std::ios::end);
	const auto total_bytes = w_stream.tellg();
	w_stream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != total_bytes) {
		std::string error_str("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(total_bytes / (sizeof(float))) + std::string(", Image width: ") + std::to_string(cols) + std::string(", Image height: ") + std::to_string(rows));
		throw std::exception(error_str.c_str());
	}
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[0]), total_bytes);
	this->w = af::transpose(af::array(cols, rows, w_ptr.get()));
}

//helper method to copy an OpenCL buffer into an OpenCL Image (fast copy that happens in the device)
cl::Image2D WatermarkFunctions::copyBufferToImage(const cl_mem *image_buff, const dim_t rows, const dim_t cols) {
	cl::Image2D image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), cols, rows, 0, NULL);
	const size_t orig[] = { 0,0,0 };
	const size_t des[] = { static_cast<size_t>(cols), static_cast<size_t>(rows), 1 };
	clEnqueueCopyBufferToImage(queue(), *image_buff, image2d(), 0, orig, des, NULL, NULL, NULL);
	return image2d;
}

//compute custom mask. supports simple kernels that just apply a mask per-pixel without needing any other configuration
void WatermarkFunctions::compute_custom_mask(const af::array& image, af::array& m)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	//fix for OpenCL 1.2 limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (cols and rows)
	const auto pad_rows = (rows % 16 == 0) ? rows : rows + 16 - (rows % 16);
	const auto pad_cols = (cols % 16 == 0) ? cols : cols + 16 - (cols % 16);
	const af::array image_transpose = image.T();
	try {
		cl_mem *image_buff = image_transpose.device<cl_mem>();
		cl::Image2D image2d = copyBufferToImage(image_buff, rows, cols);
		cl::Buffer buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * rows * cols, NULL);
		cl::Kernel kernel = cl::Kernel(program_custom, custom_kernel_name.c_str());
		kernel.setArg(0, image2d);
		kernel.setArg(1, buff);
		kernel.setArg(2, p);
		kernel.setArg(3, pad);
		queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(pad_rows, pad_cols), cl::NDRange(16, 16));
		queue.finish();
		m = afcl::array(rows, cols, buff(), af::dtype::f32, true);
		image_transpose.unlock();
	}
	catch (const cl::Error& ex) {
		std::string error_str("ERROR in compute_nvf_mask(): " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
		throw std::exception(error_str.c_str());
	}
}

af::array WatermarkFunctions::make_and_add_watermark(float& a, const std::function<void(const af::array&, af::array&, af::array&)>& compute_mask)
{
	af::array m, error_sequence;
	compute_mask(image, m, error_sequence);
	const af::array u = m * w;
	const float divisor = std::sqrt(af::sum<float>(af::pow(u, 2)) / (image.elements()));
	a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
 	return image + (a * u);
}

//public method called from host to apply the custom mask and return the watermarked image
af::array WatermarkFunctions::make_and_add_watermark_custom(float& a)
{
	return make_and_add_watermark(a, [&](const af::array& image, af::array& m, af::array& error_sequence) {
		compute_custom_mask(image, m);
	});
}

//public method called from host to apply the prediction error mask and return the watermarked image
af::array WatermarkFunctions::make_and_add_watermark_prediction_error(af::array& coefficients, float& a)
{
	return make_and_add_watermark(a, [&](const af::array& image, af::array& m, af::array& error_sequence) {
		compute_prediction_error_mask(image, m, error_sequence, coefficients, true);
	});
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
void WatermarkFunctions::compute_prediction_error_mask(const af::array& image, af::array& m_e, af::array& error_sequence, af::array& coefficients, const bool mask_needed)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	//fix for OpenCL 1.2 limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (cols)
	const auto pad_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	const af::array image_transpose = image.T();
	cl_int err;
	try {
		cl_mem *buffer = image_transpose.device<cl_mem>();
		cl::Image2D image2d = copyBufferToImage(buffer, rows, cols);
		cl::Buffer Rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * pad_cols * rows, NULL, &err);
		cl::Buffer rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * pad_cols * rows, NULL, &err);
		cl::Kernel kernel = cl::Kernel(program_me, "me", &err);
		kernel.setArg(0, image2d);
		kernel.setArg(1, Rx_buff);
		kernel.setArg(2, rx_buff);
		kernel.setArg(3, cl::Local(sizeof(float) * 4096));
		kernel.setArg(4, cl::Local(sizeof(float) * 512));
		queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(rows, pad_cols), cl::NDRange(1, 64));
		//enqueue the calculation of neighbors (x_) array before waiting "me" kernel to finish, may help a bit
		af::array x_all = af::unwrap(image, p, p, 1, 1, pad, pad, false);
		af::array x_ = af::join(1, x_all(af::span, af::seq(0, (p_squared / 2) - 1)), x_all(af::span, af::seq((p_squared / 2) + 1, af::end)));
		queue.finish(); 
		image_transpose.unlock();
		af::array Rx_partial_sums = af::moddims(afcl::array(pad_cols, rows, Rx_buff(), af::dtype::f32, true), p_squared_minus_one_squared, (pad_cols * rows) / p_squared_minus_one_squared);
		af::array rx_partial_sums = af::moddims(afcl::array(pad_cols, rows, rx_buff(), af::dtype::f32, true), p_squared_minus_one, (pad_cols * rows) / p_squared_minus_one);
		//reduction sum of blocks
		//all [p^2-1,1] blocks will be summed in rx
		//all [p^2-1, p^2-1] blocks will be summed in Rx
		af::array Rx = af::moddims(af::sum(Rx_partial_sums, 1), p_squared_minus_one, p_squared_minus_one);
		af::array rx = af::sum(rx_partial_sums, 1);
		coefficients = af::solve(Rx, rx);
		error_sequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
		if (mask_needed) {
			af::array error_sequence_abs = af::abs(error_sequence);
			m_e = error_sequence_abs / af::max<float>(error_sequence_abs);
			//display_array(m_e);
		}
	}
	catch (const cl::Error &ex) {
		std::string error_str("ERROR in compute_me_mask(): " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
		throw std::exception(error_str.c_str());
	}
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array WatermarkFunctions::calculate_error_sequence(const af::array& u, const af::array& coefficients) {
	af::array u_neighb = af::unwrap(u, p, p, 1, 1, pad, pad, false);
	af::array x_ = af::join(1, u_neighb(af::span, af::seq(0, (p_squared / 2) - 1)), u_neighb(af::span, af::seq((p_squared / 2) + 1, af::end)));
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, x_), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
void WatermarkFunctions::compute_prediction_error_mask(const af::array& image, const af::array& coeficcients, af::array& m_e, af::array& error_sequence)
{
	error_sequence = calculate_error_sequence(image, coeficcients);
	af::array error_sequence_abs = af::abs(error_sequence);
	m_e = error_sequence_abs / af::max<float>(error_sequence_abs);
}

//fast prediction error sequence calculation by using a supplied prediction filter (calls helper method)
af::array WatermarkFunctions::compute_error_sequence(const af::array& u, const af::array& coefficients)
{
	return calculate_error_sequence(u, coefficients);
}

//helper method used in detectors
float WatermarkFunctions::calculate_correlation(const af::array& e_u, const af::array& e_z) {
	float dot_ez_eu, d_ez, d_eu;
	dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() needs vectors, so we flatten the arrays
	d_ez = std::sqrt(af::sum<float>(af::pow(e_z, 2)));
	d_eu = std::sqrt(af::sum<float>(af::pow(e_u, 2)));
	return dot_ez_eu / (d_ez * d_eu);
}

//the main mask detector function
float WatermarkFunctions::mask_detector(const af::array& image, const std::function<void(const af::array&, af::array&)>& compute_custom_mask)
{
	af::array m, e_z, a_z;
	if (compute_custom_mask != nullptr) {
		compute_prediction_error_mask(image, m, e_z, a_z, false);
		compute_custom_mask(image, m);
	}
	else {
		compute_prediction_error_mask(image, m, e_z, a_z, true);
	}
	af::array u = m * w;
	af::array e_u = compute_error_sequence(u, a_z);
	return calculate_correlation(e_u, e_z);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (coefficients, x_ are supplied)
float WatermarkFunctions::mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients)
{
	af::array m_e, e_z, m_eu, e_u, a_u;
	compute_prediction_error_mask(watermarked_image, coefficients, m_e, e_z);
	af::array u = m_e * w;
	compute_prediction_error_mask(u, m_eu, e_u, a_u, false);
	return calculate_correlation(e_u, e_z);
}

//calls main mask detector for custom masks
float WatermarkFunctions::mask_detector_custom(const af::array& watermarked_image) {
	return mask_detector(watermarked_image, [&](const af::array& watermarked_image,af::array& m){
		compute_custom_mask(watermarked_image, m);
	});
}

//calls main mask detector for prediction error mask
float WatermarkFunctions::mask_detector_prediction_error(const af::array& watermarked_image) {
	return mask_detector(watermarked_image, nullptr);
}

//helper method to display an af::array in a window
void WatermarkFunctions::display_array(const af::array& array, const int width, const int height) {
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}