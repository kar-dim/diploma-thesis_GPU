#include "cuda_utils.hpp"
#include "kernels.cuh"
#include "Watermark.cuh"
#include <af/cuda.h>
#include <arrayfire.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using std::string;
using std::cout;

//constructor without specifying input image yet, it must be supplied later by calling the appropriate public method
Watermark::Watermark(const string &w_file_path, const int p, const float psnr)
	:w_file_path(w_file_path), p(p), p_squared(p* p), p_squared_minus_one(p_squared - 1), p_squared_minus_one_squared(p_squared_minus_one* p_squared_minus_one), pad(p / 2), psnr(psnr) {
	rows = -1;
	cols = -1;
	af_cuda_stream = afcu::getStream(afcu::getNativeId(af::getDevice()));
	cudaStreamCreate(&custom_kernels_stream);

}

//full constructor
Watermark::Watermark(const af::array &rgb_image, const af::array& image, const string &w_file_path, const int p, const float psnr)
	:Watermark::Watermark(w_file_path, p, psnr) {
	this->rgb_image = rgb_image;
	load_image(image);
	load_W(rows, cols);
}

Watermark::~Watermark()
{
	cudaStreamDestroy(custom_kernels_stream);
}

//supply the input image to apply watermarking and detection
void Watermark::load_image(const af::array& image) {
	this->image = image;
	this->rows = image.dims(0);
	this->cols = image.dims(1);
}

//helper method to load the random noise matrix W from the file specified.
void Watermark::load_W(const dim_t rows, const dim_t cols) {
	std::ifstream w_stream(w_file_path.c_str(), std::ios::binary);
	if (!w_stream.is_open())
		throw std::runtime_error(string("Error opening '" + w_file_path + "' file for Random noise W array\n"));
	w_stream.seekg(0, std::ios::end);
	const auto total_bytes = w_stream.tellg();
	w_stream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != total_bytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(total_bytes / (sizeof(float))) + ", Image width: " + std::to_string(cols) + ", Image height: " + std::to_string(rows) + "\n"));
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[0]), total_bytes);
	this->w = af::transpose(af::array(cols, rows, w_ptr.get()));
}

//helper method to copy an arrayfire cuda buffer into a cuda Texture Object Image (fast copy that happens in the device)
std::pair<cudaTextureObject_t, cudaArray*> Watermark::copy_array_to_texture_data(const af::array & array, const unsigned int rows, const unsigned int cols) {
	
	cudaArray* cuArray = cuda_utils::cudaMallocArray(cols, rows);
	cudaMemcpy2DToArray(cuArray, 0, 0, array.device<float>(), cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
	cudaResourceDesc resDesc = cuda_utils::createResourceDescriptor(cuArray);
	cudaTextureDesc texDesc = cuda_utils::createTextureDescriptor();
	cudaTextureObject_t texObj = cuda_utils::createTextureObject(resDesc, texDesc);
	return std::make_pair(texObj, cuArray);
}

//helper method for cleanup and to execute common tasks after the masking kernels are executed
void Watermark::synchronize_and_cleanup_texture_data(const std::pair<cudaTextureObject_t, cudaArray*> &texture_data, const af::array &array_to_unlock) {
	cudaDeviceSynchronize();
	cudaDestroyTextureObject(texture_data.first);
	cudaFreeArray(texture_data.second);
	array_to_unlock.unlock();
}

//compute custom mask. supports simple kernels that just apply a mask per-pixel without needing any other configuration
af::array Watermark::compute_custom_mask(const af::array& image)
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const af::array image_transpose = image.T();
	auto texture_data = copy_array_to_texture_data(image_transpose, rows, cols);
	float* mask_output = cuda_utils::cudaMallocPtr(rows * cols);
	auto dimensions = std::make_pair(cuda_utils::grid_size_calculate(dim3(32, 32), rows, cols), dim3(32, 32));
	nvf <<<dimensions.first, dimensions.second, 0, af_cuda_stream >>> (texture_data.first, mask_output, p*p, pad, cols, rows);
	synchronize_and_cleanup_texture_data(texture_data, image_transpose);
	return af::array(rows, cols, mask_output, afDevice);
}

//helper method to calculate the neighbors ("x_" array)
af::array Watermark::calculate_neighbors_array(const af::array& array, const int p, const int p_squared, const int pad) {
	af::array array_unwrapped = af::unwrap(array, p, p, 1, 1, pad, pad, false);
	return af::join(1, array_unwrapped(af::span, af::seq(0, (p_squared / 2) - 1)), array_unwrapped(af::span, af::seq((p_squared / 2) + 1, af::end)));
}

//helper method to sum the incomplete Rx_partial and rx_partial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int padded_cols) {
	af::array Rx_partial_sums = af::moddims(Rx_partial, p_squared_minus_one_squared, (padded_cols * rows) / p_squared_minus_one_squared);
	af::array rx_partial_sums = af::moddims(rx_partial, p_squared_minus_one, (padded_cols * rows) / p_squared_minus_one);
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	af::array Rx = af::moddims(af::sum(Rx_partial_sums, 1), p_squared_minus_one, p_squared_minus_one);
	af::array rx = af::sum(rx_partial_sums, 1);
	return std::make_pair(Rx, rx);
}

af::array Watermark::make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE mask_type, IMAGE_TYPE image_type)
{
	af::array mask, error_sequence;
	if (mask_type == MASK_TYPE::ME)
		mask = compute_prediction_error_mask(image, error_sequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
	else
		mask = compute_custom_mask(image);
	const af::array u = mask * w;
	const float divisor = std::sqrt(af::sum<float>(af::pow(u, 2)) / (image.elements()));
	a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
	return image_type == IMAGE_TYPE::RGB ?
		af::clamp(rgb_image + af::tile((u * a), 1, 1, static_cast<unsigned int>(image.dims(2))), 0, 255) :
		af::clamp(image + (u * a), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
af::array Watermark::compute_prediction_error_mask(const af::array& image, af::array& error_sequence, af::array& coefficients, const bool mask_needed)
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const af::array image_transpose = image.T();
	const auto padded_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	//copy arrayfire array from device to device's texture cache and allocate Rx,rx buffers
	auto texture_data = copy_array_to_texture_data(image_transpose, rows, cols);
	float* Rx_buff = cuda_utils::cudaMallocPtr(rows * padded_cols);
	float* rx_buff = cuda_utils::cudaMallocPtr(rows * padded_cols);
	//call custom kernel to fill Rx and rx partial sums (in different stream than arrayfire, may help)
	auto dimensions = std::make_pair(cuda_utils::grid_size_calculate(dim3(1, 64), rows, padded_cols), dim3(1, 64));
	me_p3 <<<dimensions.first, dimensions.second, 0, custom_kernels_stream>>> (texture_data.first, Rx_buff, rx_buff, cols, padded_cols, rows);
	//calculate the neighbors "x_" array and wait for custom kernel to finish and release texture memory
	af::array x_ = calculate_neighbors_array(image, p, p_squared, pad);
	synchronize_and_cleanup_texture_data(texture_data, image_transpose);
	//transform the partial Rx,rx arrays by summing and changing their dimensions
	const auto correlation_arrays = correlation_arrays_transformation(af::array(padded_cols, rows, Rx_buff, afDevice), af::array(padded_cols, rows, rx_buff, afDevice), padded_cols);
	//solve the system to get coefficients and error sequence, and optionally the mask if needed
	coefficients = af::solve(correlation_arrays.first, correlation_arrays.second);
	error_sequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
	if (mask_needed) {
		const af::array error_sequence_abs = af::abs(error_sequence);
		return error_sequence_abs / af::max<float>(error_sequence_abs);
	}
	return af::array();
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array Watermark::calculate_error_sequence(const af::array& u, const af::array& coefficients) {
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, calculate_neighbors_array(u, p, p_squared, pad)), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
af::array Watermark::compute_prediction_error_mask(const af::array& image, const af::array& coeficcients, af::array& error_sequence)
{
	error_sequence = calculate_error_sequence(image, coeficcients);
	const af::array error_sequence_abs = af::abs(error_sequence);
	return error_sequence_abs / af::max<float>(error_sequence_abs);
}

//helper method used in detectors
float Watermark::calculate_correlation(const af::array& e_u, const af::array& e_z) {
	double dot_ez_eu = af::dot<double>(af::flat(e_u), af::flat(e_z)); //dot() needs vectors, so we flatten the arrays
	return static_cast<float>(dot_ez_eu / (af::norm(e_z) * af::norm(e_u)));
}

//the main mask detector function
float Watermark::mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type)
{
	af::array mask, e_z, a_z;
	if (mask_type == MASK_TYPE::NVF) {
		compute_prediction_error_mask(watermarked_image, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_NO);
		mask = compute_custom_mask(watermarked_image);
	}
	else {
		mask = compute_prediction_error_mask(watermarked_image, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	}
	const af::array u = mask * w;
	const af::array e_u = calculate_error_sequence(u, a_z);
	return calculate_correlation(e_u, e_z);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (coefficients, x_ are supplied)
float Watermark::mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients)
{
	af::array m_e, e_z, e_u, a_u;
	m_e = compute_prediction_error_mask(watermarked_image, coefficients, e_z);
	const af::array u = m_e * w;
	compute_prediction_error_mask(u, e_u, a_u, ME_MASK_CALCULATION_REQUIRED_NO);
	return calculate_correlation(e_u, e_z);
}

//helper method to display an af::array in a window
void Watermark::display_array(const af::array& array, const int width, const int height) {
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}