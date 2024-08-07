__kernel void nvf(__read_only image2d_t image, 
				  __global float* m_nvf, 
				    const int p,
					const int pad)
{	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	const int width = get_image_width(image), height = get_image_height(image);
	const int x = get_global_id(1), y = get_global_id(0);

	//fix for OpenCL 1.2 where global size % local size should be 0, and local size is padded, a bound check is needed
	if (y < height && x < width) {
		int i, j, k = 0;
		float mean = 0.0f, variance = 0.0f;
		//p_squared is supplied at compile-time to minimize array elements (VLAs not supported in OpenCL)
		float local_values[p_squared];
		for (j = x - pad; j <= x + pad; j++) {
			for (i = y - pad; i <= y + pad; i++) {
				local_values[k] = read_imagef(image, sampler, (int2)(j, i)).x;
				mean += local_values[k];
				k++;
			}
		}
		mean /= p_squared;
		for (i = 0; i < p_squared; i++)
			variance += pown(local_values[i] - mean, 2);
		//calculate mask
		const float nvf_mask = variance / ((p_squared - 1) + variance);
		//write pixel value
		m_nvf[(x * height) + y] = nvf_mask;
	}
}