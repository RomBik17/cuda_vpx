
#include "inter_cuda_kernel.cuh"

extern "C" int wrap_cuda_inter_prediction(int n, double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf, 
	VP9_COMMON * cm, VP9Decoder * pbi, int tile_rows, int tile_cols)
{
	cuda_inter_prediction(n, gpu_copy, gpu_run, size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols);
	return 0;
}