#pragma once

#ifdef WATERMARK_EXPORT
#define WATERMARK_API __declspec(dllexport)
#else
#define WATERMARK_API __declspec(dllimport)
#endif

#include <arrayfire.h>
#include <cuda_runtime.h>
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
class WATERMARK_API Watermark {
private:
	char* w_file_path;
	const int p, p_squared, p_squared_minus_one, p_squared_minus_one_squared, pad;
	const float psnr;
	af::array rgb_image, image, w;
	dim_t rows, cols;
	cudaStream_t af_cuda_stream, custom_kernels_stream;

	float calculate_correlation(const af::array& e_u, const af::array& e_z) const;
	af::array compute_custom_mask(const af::array& image) const;
	af::array compute_prediction_error_mask(const af::array& image, af::array& error_sequence, af::array& coefficients, const bool mask_needed) const;
	af::array compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& error_sequence) const;
	af::array calculate_error_sequence(const af::array& u, const af::array& coefficients) const;
	af::array calculate_neighbors_array(const af::array& array, const int p, const int p_squared, const int pad) const;
	std::pair<af::array, af::array> correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int padded_cols) const;
	void synchronize_and_cleanup_texture_data(const std::pair<cudaTextureObject_t, cudaArray*>& texture_data, const af::array& array_to_unlock) const;
	std::pair<cudaTextureObject_t, cudaArray*> copy_array_to_texture_data(const af::array &image, const unsigned int rows, const unsigned int cols) const;
public:
	Watermark(const af::array& rgb_image, const af::array& image, const std::string& w_file_path, const int p, const float psnr);
	Watermark(const std::string &w_file_path, const int p, const float psnr);
	~Watermark();
	void load_W(const dim_t rows, const dim_t cols);
	void load_image(const af::array& image);
	af::array make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE type, IMAGE_TYPE image_type) const;
	float mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type) const;
	float mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients) const;
	static void display_array(const af::array& array, const int width = 1600, const int height = 900);
};