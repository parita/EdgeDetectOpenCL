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
//#define pi 3.14159265
__kernel void func(__global const float* ckernel,__global const float* ckernel2, __global const float* in_img, 
                       __global float* mag,__global float* ang, const unsigned int k, 
                        const unsigned int h, const unsigned int w,
                        __local float* ckernel_block,__local float* ckernel2_block, __local float* img_block, 
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
        ckernel_block[lrow*k+lcol]  = ckernel[lrow*k+lcol];
        ckernel2_block[lrow*k+lcol] = ckernel2[lrow*k+lcol];
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
    float tmp  = 0.0;
    float tmp2 = 0.0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
        {
            tmp += ckernel_block[i*k+j]*img_block[(lrow+i)*img_w+(lcol+j)];
            tmp2 += ckernel2_block[i*k+j]*img_block[(lrow+i)*img_w+(lcol+j)];
        }
               
    float angle=0.0;
    float m=0.0;
    if (row < h && col < w)
    {
        m = sqrt(tmp*tmp+tmp2*tmp2);
        mag[row*w+col] = m;
        angle = (180/M_PI)*acos(tmp/m);
        
        if (angle >= 0.0 && angle < 22.5)
            angle = 0;
        else if (angle >= 22.5 && angle < 67.5)
            angle = 45;
        else if (angle >= 67.5 && angle < 112.5)
            angle = 90;
        else if (angle >= 112.5 && angle < 157.5)
            angle = 135;
        else
            angle = 180;
        
        ang[row*w+col] = angle;
    }
}
"""

threshold_kernel = """
__kernel void func(__global const float* src_img, __global float* out_img, 
                   __global const float* orientation, const float thres, const unsigned int R, 
                    const unsigned int C)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    float centre = 0.0;
    float left = 0.0;
    float right = 0.0;
    if (i < R && j < C)
    {
        int angle = orientation[i*C + j];
        switch(angle)
        {
            case 0: if (j > 0)  left = src_img[i*C + (j-1)];
                    if (j < C-1) right = src_img[i*C + (j+1)];
                    break;
            case 45: if (i < R-1 && j > 0) left = src_img[(i+1)*C + (j-1)];
                     if (i > 0 && j < C-1) right = src_img[(i-1)*C + (j+1)];
                     break;
            case 90: if (i > 0) left = src_img[(i-1)*C + j];
                     if (i < R-1) right = src_img[(i+1)*C + j];
                     break;
            case 135: if(i > 0 && j > 0) left = src_img[(i-1)*C + (j-1)];
                      if(i < R-1 && j < C-1) right = src_img[(i+1)*C + (j+1)];
                      break;
            default: break;
        }
        centre = src_img[i*C + j];
        if (centre > left && centre > right && centre > thres)
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

