快手网红爆料视频免费下载快手抖音网红爆料

   }

    for (int num = 1; num <= MAX_NUM_STREAMS; num *= 2)
    {
        timing(h_x, h_y, h_z, d_x, d_y, d_z, num);
    }

    for (int i = 0 ; i < MAX_NUM_STREAMS; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaFreeHost(h_x));
    CHECK(cudaFreeHost(h_y));
    CHECK(cudaFreeHost(h_z));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    return 0;
}

void __global__ add(const real *x, const real *y, real *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        for (int i = 0; i < 40; ++i)
        {
            z[n] = x[n] + y[n];
