
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/decoder/vp9_decoder.h"
#include "buffers_struct.h"

#define yv12_align_addr(addr, align) \
  (void *)(((size_t)(addr) + ((align)-1)) & (size_t) - (align))

#define MAX(a, b) ((a > b) ? a : b)

const int threadsPerBlock = 512;

#define FILTER_BITS 7

#define SUBPEL_BITS 4
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define SUBPEL_SHIFTS (1 << SUBPEL_BITS)
#define SUBPEL_TAPS 8

#define REF_SCALE_SHIFT 14
#define REF_NO_SCALE (1 << REF_SCALE_SHIFT)
#define REF_INVALID_SCALE (-1)

typedef int16_t InterpKernel[SUBPEL_TAPS];

__constant__ DECLARE_ALIGNED(256, static InterpKernel,
cuda_bilinear_filters[SUBPEL_SHIFTS]) = {
{ 0, 0, 0, 128, 0, 0, 0, 0 },  { 0, 0, 0, 120, 8, 0, 0, 0 },
{ 0, 0, 0, 112, 16, 0, 0, 0 }, { 0, 0, 0, 104, 24, 0, 0, 0 },
{ 0, 0, 0, 96, 32, 0, 0, 0 },  { 0, 0, 0, 88, 40, 0, 0, 0 },
{ 0, 0, 0, 80, 48, 0, 0, 0 },  { 0, 0, 0, 72, 56, 0, 0, 0 },
{ 0, 0, 0, 64, 64, 0, 0, 0 },  { 0, 0, 0, 56, 72, 0, 0, 0 },
{ 0, 0, 0, 48, 80, 0, 0, 0 },  { 0, 0, 0, 40, 88, 0, 0, 0 },
{ 0, 0, 0, 32, 96, 0, 0, 0 },  { 0, 0, 0, 24, 104, 0, 0, 0 },
{ 0, 0, 0, 16, 112, 0, 0, 0 }, { 0, 0, 0, 8, 120, 0, 0, 0 }
};

// Lagrangian interpolation filter
__constant__ DECLARE_ALIGNED(256, static InterpKernel,
cuda_sub_pel_filters_8[SUBPEL_SHIFTS]) = {
{ 0, 0, 0, 128, 0, 0, 0, 0 },        { 0, 1, -5, 126, 8, -3, 1, 0 },
{ -1, 3, -10, 122, 18, -6, 2, 0 },   { -1, 4, -13, 118, 27, -9, 3, -1 },
{ -1, 4, -16, 112, 37, -11, 4, -1 }, { -1, 5, -18, 105, 48, -14, 4, -1 },
{ -1, 5, -19, 97, 58, -16, 5, -1 },  { -1, 6, -19, 88, 68, -18, 5, -1 },
{ -1, 6, -19, 78, 78, -19, 6, -1 },  { -1, 5, -18, 68, 88, -19, 6, -1 },
{ -1, 5, -16, 58, 97, -19, 5, -1 },  { -1, 4, -14, 48, 105, -18, 5, -1 },
{ -1, 4, -11, 37, 112, -16, 4, -1 }, { -1, 3, -9, 27, 118, -13, 4, -1 },
{ 0, 2, -6, 18, 122, -10, 3, -1 },   { 0, 1, -3, 8, 126, -5, 1, 0 }
};

// DCT based filter
__constant__ DECLARE_ALIGNED(256, static InterpKernel,
cuda_sub_pel_filters_8s[SUBPEL_SHIFTS]) = {
{ 0, 0, 0, 128, 0, 0, 0, 0 },         { -1, 3, -7, 127, 8, -3, 1, 0 },
{ -2, 5, -13, 125, 17, -6, 3, -1 },   { -3, 7, -17, 121, 27, -10, 5, -2 },
{ -4, 9, -20, 115, 37, -13, 6, -2 },  { -4, 10, -23, 108, 48, -16, 8, -3 },
{ -4, 10, -24, 100, 59, -19, 9, -3 }, { -4, 11, -24, 90, 70, -21, 10, -4 },
{ -4, 11, -23, 80, 80, -23, 11, -4 }, { -4, 10, -21, 70, 90, -24, 11, -4 },
{ -3, 9, -19, 59, 100, -24, 10, -4 }, { -3, 8, -16, 48, 108, -23, 10, -4 },
{ -2, 6, -13, 37, 115, -20, 9, -4 },  { -2, 5, -10, 27, 121, -17, 7, -3 },
{ -1, 3, -6, 17, 125, -13, 5, -2 },   { 0, 1, -3, 8, 127, -7, 3, -1 }
};

// freqmultiplier = 0.5
__constant__ DECLARE_ALIGNED(256, static InterpKernel,
cuda_sub_pel_filters_8lp[SUBPEL_SHIFTS]) = {
{ 0, 0, 0, 128, 0, 0, 0, 0 },       { -3, -1, 32, 64, 38, 1, -3, 0 },
{ -2, -2, 29, 63, 41, 2, -3, 0 },   { -2, -2, 26, 63, 43, 4, -4, 0 },
{ -2, -3, 24, 62, 46, 5, -4, 0 },   { -2, -3, 21, 60, 49, 7, -4, 0 },
{ -1, -4, 18, 59, 51, 9, -4, 0 },   { -1, -4, 16, 57, 53, 12, -4, -1 },
{ -1, -4, 14, 55, 55, 14, -4, -1 }, { -1, -4, 12, 53, 57, 16, -4, -1 },
{ 0, -4, 9, 51, 59, 18, -4, -1 },   { 0, -4, 7, 49, 60, 21, -3, -2 },
{ 0, -4, 5, 46, 62, 24, -3, -2 },   { 0, -4, 4, 43, 63, 26, -2, -2 },
{ 0, -3, 2, 41, 63, 29, -2, -2 },   { 0, -3, 1, 38, 64, 32, -1, -3 }
};

// 4-tap filter
__constant__ DECLARE_ALIGNED(256, static InterpKernel,
cuda_sub_pel_filters_4[SUBPEL_SHIFTS]) = {
{ 0, 0, 0, 128, 0, 0, 0, 0 },     { 0, 0, -4, 126, 8, -2, 0, 0 },
{ 0, 0, -6, 120, 18, -4, 0, 0 },  { 0, 0, -8, 114, 28, -6, 0, 0 },
{ 0, 0, -10, 108, 36, -6, 0, 0 }, { 0, 0, -12, 102, 46, -8, 0, 0 },
{ 0, 0, -12, 94, 56, -10, 0, 0 }, { 0, 0, -12, 84, 66, -10, 0, 0 },
{ 0, 0, -12, 76, 76, -12, 0, 0 }, { 0, 0, -10, 66, 84, -12, 0, 0 },
{ 0, 0, -10, 56, 94, -12, 0, 0 }, { 0, 0, -8, 46, 102, -12, 0, 0 },
{ 0, 0, -6, 36, 108, -10, 0, 0 }, { 0, 0, -6, 28, 114, -8, 0, 0 },
{ 0, 0, -4, 18, 120, -6, 0, 0 },  { 0, 0, -2, 8, 126, -4, 0, 0 }
};

__constant__ InterpKernel* cuda_vp9_filter_kernels[5];

__constant__ int idxFiltKern[1];

__global__ void copySF(scale_factors* cuda_sf, int x_scale_fp, int y_scale_fp);

__host__ int cuda_inter_prediction(int n, double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
	VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols);

__global__ void setFK();
