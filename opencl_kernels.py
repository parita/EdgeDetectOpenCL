convolution_kernel = """
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define BLOCK_DIM 32
__kernel void func(__global const float* ckernel, __global const float* in_img, 
                        __global float* out_img, const unsigned int k, 
                        const unsigned int h, const unsigned int w,
                        __local float* ckernel_block, __local float* img_block)
{
    int img_w = BLOCK_DIM + k; 
    unsigned int lrow = get_local_id(0); 
    unsigned int lcol = get_local_id(1);
    unsigned int wgrow = get_group_id(0);
    unsigned int wgcol = get_group_id(1);
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);
    if (lrow < k && lcol < k)
    {
        ckernel_block[lrow*k+lcol] = ckernel[lrow*k+lcol];
    }
    if (row < h && col < w)
    {
        img_block[lrow*img_w + lcol] = in_img[row*(w+k-1) + col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lrow < k & lcol < k)
    {
        img_block[(BLOCK_DIM+lrow)*img_w + (BLOCK_DIM+lcol)] = 
                        in_img[(BLOCK_DIM+row)*(w+k-1) + (BLOCK_DIM+col)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lrow < BLOCK_DIM && lcol < k)
        img_block[lrow*img_w + (BLOCK_DIM+lcol)] = in_img[row*(w+k-1) + (BLOCK_DIM+col)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lcol < BLOCK_DIM && lrow < k)
        img_block[(BLOCK_DIM+lrow)*img_w + lcol] = in_img[(BLOCK_DIM+row)*(w+k-1) + col];
    barrier(CLK_LOCAL_MEM_FENCE);

    float tmp = 0.0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            tmp += ckernel_block[i*k+j]*img_block[(lrow+i)*img_w+(lcol+j)];
    
    out_img[row*w + col] = tmp;
}
"""
