__kernel void calculate_neighbors_p3(
    __read_only image2d_t image, 
    __global float* x_)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    const int width = get_image_width(image), height = get_image_height(image);
    const int outputIndex = (x * height + y);

    if (x < width && y < height) 
    {
        // Write 8 neighboring pixels into global memory (coalesced writes)
        x_[0 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x - 1, y - 1)).x;
        x_[1 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x - 1, y)).x;
        x_[2 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x - 1, y + 1)).x;
        x_[3 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x, y - 1)).x;
        x_[4 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x, y + 1)).x;
        x_[5 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x + 1, y - 1)).x;
        x_[6 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x + 1, y)).x;
        x_[7 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(x + 1, y + 1)).x;
    }
}