#include "kernels.cuh"

__global__ void nvf(cudaTextureObject_t texObj, float* m_nvf, const int p_squared, const int pad, const int width, const int height) 
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) 
        return;

	int i, j, k = 0;
	float mean = 0.0f, variance = 0.0f, local_mean_diff;
	//maximum local values size is 81 for a 9x9 block
	float local_values[81];
	for (j = x - pad; j <= x + pad; j++) {
		for (i = y - pad; i <= y + pad; i++) {
			local_values[k] = tex2D<float>(texObj, j, i);
			mean += local_values[k];
			k++;
		}
	}
	mean /= p_squared;
	for (i = 0; i < p_squared; i++) {
        local_mean_diff = local_values[i] - mean;
        variance += local_mean_diff * local_mean_diff;
	}
	//calculate mask and write pixel value
    const float nvf_mask = variance / ((p_squared - 1) + variance);
	m_nvf[(x * height) + y] = nvf_mask;
}

__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const int width, const int padded_width, const int height) 
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
	const int local_id = threadIdx.y;
	const int rx_stride = local_id * 8;
	const int output_index = (y * padded_width) + x;
    const bool is_padded = padded_width > width;

    __shared__ float Rx_local[64][36];
    __shared__ float rx_local[512];

    if (y >= height)
        return;

    if (x < width) {
        int counter = 0;
        float x_[9];
        for (int j = x - 1; j <= x + 1; j++)
            for (int i = y - 1; i <= y + 1; i++)
                x_[counter++] = tex2D<float>(texObj, j, i);
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 64 local Rx and 8 local rx values
        counter = 0;
        for (int i = 0; i < 8; i++) {
            rx_local[rx_stride + i] = x_[i] * current_pixel;
            Rx_local[local_id][counter] = x_[i] * x_[i];
            counter++;
            for (int j = i + 1; j < 8; j++) {
                Rx_local[local_id][counter] = x_[i] * x_[j];
                counter++;
            }
        }
    }

    //each thread will calculate the reduction sums of Rx and rx and write them to global memory
    //if image is padded we don't want to sum the garbage local array values, we could zero the local array
    //but it would cost time, instead it is better to calculate what is needed directly
    __syncthreads();
    const int limit = (is_padded && padded_width - x <= 64) ? 64 - (padded_width - width) : 64;
    float reduction_sum_Rx = 0.0f, reduction_sum_rx = 0.0f;
    for (int j = 0; j < limit; j++)
        reduction_sum_Rx += Rx_local[j][Rx_mappings[local_id]];
    for (int j = 0; j < (512 * limit) / 64; j += 64)
        reduction_sum_rx += rx_local[local_id + j];
    Rx[output_index] = reduction_sum_Rx;
    rx[output_index] = reduction_sum_rx; 
}