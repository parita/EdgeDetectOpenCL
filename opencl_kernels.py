convolution_kernel = """
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
//#define BLOCK_DIM 32
__kernel void func(__global const float* ckernel, __global const float* in_img, 
                        __global float* out_img, const unsigned int k, 
                        const unsigned int h, const unsigned int w,
                        __local float* ckernel_block, __local float* img_block, 
                        const unsigned int BLOCK_DIM)
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
    
    if (row < h && col < w)
        out_img[row*w + col] = tmp;
}
"""

gradient_kernel = """
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
//#define BLOCK_DIM 32
__kernel void func(__global const float* ckernel, __global const float* in_img, 
                        __global float* Gx, __global float* Gy,
                        __global float* out_img, const unsigned int k, 
                        const unsigned int h, const unsigned int w,
                        __local float* ckernel_blockx, __local float* ckernel_blocky,
                        __local float* img_block, 
                        const unsigned int BLOCK_DIM)
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
        ckernel_blockx[lrow*k+lcol] = ckernel[lrow*k+lcol];
        ckernel_blocky[lrow*k+lcol] = ckernel[lcol*k+lrow];
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

    float tmpx = 0.0;
    float tmpy = 0.0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
    {
            tmpx += ckernel_blockx[i*k+j]*img_block[(lrow+i)*img_w+(lcol+j)];
            tmpx += ckernel_blocky[i*k+j]*img_block[(lrow+i)*img_w+(lcol+j)];
    }
    
    if (row < h && col < w)
    {
        Gx[row*w + col] = tmpx;
        Gy[row*w + col] = tmpy;
        out_img[row*w + col] = sqrt(tmpx*tmpx + tmpy*tmpy);
    }
}
"""

threshold_kernel = """
__kernel void func(__global const float* src_img, __global float* out_img, 
                    const float thres, const unsigned int R, 
                    const unsigned int C)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    if (i < R && j < C)
    {
        if (src_img[i*C + j] > thres)
            out_img[i*C + j] = 1.0;
        else
            out_img[i*C + j] = 0.0;
    }
}
"""

laplacian_along_gradient_kernel = """
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
//#define BLOCK_DIM 32
__kernel void func(__global const float* ckernel, __global const float* in_img, 
                        __global float* out_img, __global const float* Gx,
                        __global float* Gy, const unsigned int k, 
                        const unsigned int h, const unsigned int w,
                        __local float* ckernel_block, __local float* img_block, 
                        const unsigned int BLOCK_DIM)
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
    
    if (row < h && col < w)
        out_img[row*w + col] = tmp;
        
}
"""

