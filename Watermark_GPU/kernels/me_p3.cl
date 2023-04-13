#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void me(__read_only image2d_t padded, 
				 __global half *Rx,
				 __global half *rx,
				 __global half *neighb
				)
{	
	const sampler_t sampler= CLK_NORMALIZED_COORDS_FALSE 
						| CLK_ADDRESS_NONE 
						| CLK_FILTER_NEAREST;
						
	const uint width = get_image_width(padded);
	const uint height = get_image_height(padded);
	
	const uint2 pixelcoord = (uint2) (get_global_id(0), get_global_id(1));
	
	if (pixelcoord.y <= height - 2 && pixelcoord.y >= 1 && pixelcoord.x <= width - 2 && pixelcoord.x >= 1){
		uint k=0, i, j;
		half x_[9]; 
		for (j=pixelcoord.x - 1; j<= pixelcoord.x + 1; j++){
			for (i=pixelcoord.y - 1; i<= pixelcoord.y + 1; i++){
				x_[k] = read_imageh(padded, sampler, (int2)(j,i) ).x;
				k++;
			}
		}
		const half cur_value = x_[4];
		
		for (i = 4; i < 8; i++)
            x_[i] = x_[i+1];
		
		const uint real_height = height - 2;
		const uint y_minus_pad = pixelcoord.y - 1;
		const uint x_minus_pad = pixelcoord.x - 1;
		const uint y_minus_pad_mul_8 = y_minus_pad * 8;
		const uint y_minus_pad_mul_64 = y_minus_pad_mul_8 * 8;
		const uint x_minus_pad_mul_8_mul_real_height = x_minus_pad * real_height * 8;
		const uint x_minus_pad_mul_64_mul_real_height = x_minus_pad_mul_8_mul_real_height * 8;
		
		uint counter=0;
		for (i=0; i<8; i++){
			rx[i + x_minus_pad_mul_8_mul_real_height + y_minus_pad_mul_8] = x_[i] * cur_value;
			neighb[i + x_minus_pad_mul_8_mul_real_height + y_minus_pad_mul_8] = x_[i];
			for (j=0; j<8; j++){
				Rx[counter + x_minus_pad_mul_64_mul_real_height + y_minus_pad_mul_64] = x_[i] * x_[j];
				counter=counter+1;
			}
		}
		
	}
}