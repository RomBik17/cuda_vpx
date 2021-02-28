

#include "inter_cuda_kernel.cuh"

#include "time.h"
#include <cstdio>
#include <cstring>

#include "assert.h"

__device__ static int64_t cuda_scaled_buffer_offset(int x_offset, int y_offset, int stride, const struct scale_factors* sf) {
	const int x = sf ? sf->scale_value_x(x_offset, sf) : x_offset;
	const int y = sf ? sf->scale_value_y(y_offset, sf) : y_offset;
	return (int64_t)y * stride + x;
}

__device__ static void cuda_setup_pred_plane(struct buf_2d* dst, uint8_t* src,
		int stride, int mi_row, int mi_col, const struct scale_factors* scale, int subsampling_x, int subsampling_y) {
	const int x = (MI_SIZE * mi_col) >> subsampling_x;
	const int y = (MI_SIZE * mi_row) >> subsampling_y;
	dst->buf = src + cuda_scaled_buffer_offset(x, y, stride, scale);
	dst->stride = stride;
}

__device__ void cuda_vp9_setup_pre_planes(MACROBLOCKD* xd, int idx, const YV12_BUFFER_CONFIG* src, int mi_row, int mi_col, const struct scale_factors* sf) {
	if (src != NULL) {
		int i;
		uint8_t* const buffers[MAX_MB_PLANE] = { src->y_buffer, src->u_buffer,
												 src->v_buffer };
		const int strides[MAX_MB_PLANE] = { src->y_stride, src->uv_stride,
											src->uv_stride };
		for (i = 0; i < MAX_MB_PLANE; ++i) {
			struct macroblockd_plane* const pd = &xd->plane[i];
			cuda_setup_pred_plane(&pd->pre[idx], buffers[i], strides[i], mi_row, mi_col,
				sf, pd->subsampling_x, pd->subsampling_y);
		}
	}
}

__device__ static int cuda_round_mv_comp_q2(int value) {
	return (value < 0 ? value - 1 : value + 1) / 2;
}

__device__ static int cuda_round_mv_comp_q4(int value) {
	return (value < 0 ? value - 2 : value + 2) / 4;
}

__device__ static MV cuda_mi_mv_pred_q2(const MODE_INFO* mi, int idx, int block0, int block1) {
	MV res = { cuda_round_mv_comp_q2(mi->bmi[block0].as_mv[idx].as_mv.row +
								mi->bmi[block1].as_mv[idx].as_mv.row),
			   cuda_round_mv_comp_q2(mi->bmi[block0].as_mv[idx].as_mv.col +
								mi->bmi[block1].as_mv[idx].as_mv.col) };
	return res;
}

__device__ static MV cuda_mi_mv_pred_q4(const MODE_INFO* mi, int idx) {
	MV res = { cuda_round_mv_comp_q4(mi->bmi[0].as_mv[idx].as_mv.row +
								mi->bmi[1].as_mv[idx].as_mv.row +
								mi->bmi[2].as_mv[idx].as_mv.row +
								mi->bmi[3].as_mv[idx].as_mv.row),
			   cuda_round_mv_comp_q4(mi->bmi[0].as_mv[idx].as_mv.col +
								mi->bmi[1].as_mv[idx].as_mv.col +
								mi->bmi[2].as_mv[idx].as_mv.col +
								mi->bmi[3].as_mv[idx].as_mv.col) };
	return res;
}

__device__ MV cuda_average_split_mvs(const struct macroblockd_plane* pd, const MODE_INFO* mi, int ref, int block) {
	const int ss_idx = ((pd->subsampling_x > 0) << 1) | (pd->subsampling_y > 0);
	MV res = { 0, 0 };
	switch (ss_idx) {
	case 0: res = mi->bmi[block].as_mv[ref].as_mv; break;
	case 1: res = cuda_mi_mv_pred_q2(mi, ref, block, block + 2); break;
	case 2: res = cuda_mi_mv_pred_q2(mi, ref, block, block + 1); break;
	case 3: res = cuda_mi_mv_pred_q4(mi, ref); break;
	default: assert(ss_idx <= 3 && ss_idx >= 0);
	}
	return res;
}

__device__ int cuda_clamp(int value, int low, int high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ static void cuda_clamp_mv(MV* mv, int min_col, int max_col, int min_row, int max_row) {
	mv->col = cuda_clamp(mv->col, min_col, max_col);
	mv->row = cuda_clamp(mv->row, min_row, max_row);
}

__device__ MV cuda_clamp_mv_to_umv_border_sb(const MACROBLOCKD* xd, const MV* src_mv, int bw, int bh, int ss_x, int ss_y) {
	const int spel_left = (VP9_INTERP_EXTEND + bw) << SUBPEL_BITS;
	const int spel_right = spel_left - SUBPEL_SHIFTS;
	const int spel_top = (VP9_INTERP_EXTEND + bh) << SUBPEL_BITS;
	const int spel_bottom = spel_top - SUBPEL_SHIFTS;
	MV clamped_mv = { (short)(src_mv->row * (1 << (1 - ss_y))),
					  (short)(src_mv->col * (1 << (1 - ss_x))) };
	assert(ss_x <= 1);
	assert(ss_y <= 1);

	cuda_clamp_mv(&clamped_mv, xd->mb_to_left_edge * (1 << (1 - ss_x)) - spel_left,
		xd->mb_to_right_edge * (1 << (1 - ss_x)) + spel_right,
		xd->mb_to_top_edge * (1 << (1 - ss_y)) - spel_top,
		xd->mb_to_bottom_edge * (1 << (1 - ss_y)) + spel_bottom);

	return clamped_mv;
}

__device__ static void* cuda_vpx_memset16(void* dest, int val, size_t length) {
	size_t i;
	uint16_t* dest16 = (uint16_t*)dest;
	for (i = 0; i < length; i++) *dest16++ = val;
	return dest;
}

__device__ int cuda_vp9_is_valid_scale(const struct scale_factors* sf)
{
	return sf->x_scale_fp != REF_INVALID_SCALE &&
		sf->y_scale_fp != REF_INVALID_SCALE;
}

__device__ int cuda_vp9_is_scaled(const struct scale_factors* sf)
{
	return cuda_vp9_is_valid_scale(sf) &&
		(sf->x_scale_fp != REF_NO_SCALE || sf->y_scale_fp != REF_NO_SCALE);
}

__device__ int unscaled_value(int val, const struct scale_factors* sf) {
	(void)sf;
	return val;
}

__device__ int scaled_x(int val, const struct scale_factors* sf)
{
	return (int)((int64_t)val * sf->x_scale_fp >> REF_SCALE_SHIFT);
}

__device__  int scaled_y(int val, const struct scale_factors* sf)
{
	return (int)((int64_t)val * sf->y_scale_fp >> REF_SCALE_SHIFT);
}

__device__ int cuda_valid_ref_frame_size(int ref_width, int ref_height, int this_width, int this_height)
{
	return 2 * this_width >= ref_width && 2 * this_height >= ref_height &&
		this_width <= 16 * ref_width && this_height <= 16 * ref_height;
}

__device__ int get_fixed_point_scale_factor(int other_size, int this_size)
{
	return (other_size << REF_SCALE_SHIFT) / this_size;
}

__device__ MV32 cuda_vp9_scale_mv(const MV* mv, int x, int y, const struct scale_factors* sf) {
	const int x_off_q4 = scaled_x(x << SUBPEL_BITS, sf) & SUBPEL_MASK;
	const int y_off_q4 = scaled_y(y << SUBPEL_BITS, sf) & SUBPEL_MASK;
	const MV32 res = { scaled_y(mv->row, sf) + y_off_q4,
					   scaled_x(mv->col, sf) + x_off_q4 };
	return res;
}

__device__ uint8_t cuda_clip_pixel(int val)
{
	return (val > 255) ? 255 : (val < 0) ? 0 : val;
}

__device__ double cuda_fclamp(double value, double low, double high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ int64_t cuda_lclamp(int64_t value, int64_t low, int64_t high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ uint16_t cuda_clip_pixel_highbd(int val, int bd)
{
	switch (bd)
	{
		case 8:
		default: return (uint16_t)cuda_clamp(val, 0, 255);
		case 10: return (uint16_t)cuda_clamp(val, 0, 1023);
		case 12: return (uint16_t)cuda_clamp(val, 0, 4095);
	}
}

__device__ static void cuda_convolve_horiz(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                           const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint8_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_convolve_avg_horiz(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                               const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint8_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = ROUND_POWER_OF_TWO(
				dst[x] + cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)), 1);
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_convolve_vert(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                          const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint8_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k)
				sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ static void cuda_convolve_avg_vert(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                              const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint8_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k)
				sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = ROUND_POWER_OF_TWO(
				dst[y * dst_stride] +
				cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)),
				1);
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ void cuda_vpx_convolve8_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                           const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)y0_q4;
	(void)y_step_q4;
	cuda_convolve_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_avg_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                               const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)y0_q4;
	(void)y_step_q4;
	cuda_convolve_avg_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                          const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)x0_q4;
	(void)x_step_q4;
	cuda_convolve_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_avg_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                              const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)x0_q4;
	(void)x_step_q4;
	cuda_convolve_avg_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                     int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	uint8_t temp[64 * 135];
	const int intermediate_height =
		(((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

	assert(w <= 64);
	assert(h <= 64);
	assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
	assert(x_step_q4 <= 64);

	cuda_convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64, filter, x0_q4, x_step_q4, w, intermediate_height);
	cuda_convolve_vert(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride, filter, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_convolve_avg_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                        int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	int x, y;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;

	for (y = 0; y < h; ++y)
	{
		for (x = 0; x < w; ++x) dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_convolve8_avg_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	// Fixed size intermediate buffer places limits on parameters.
	DECLARE_ALIGNED(16, uint8_t, temp[64 * 64]);
	assert(w <= 64);
	assert(h <= 64);

	cuda_vpx_convolve8_c(src, src_stride, temp, 64, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
	cuda_vpx_convolve_avg_c(temp, 64, dst, dst_stride, NULL, 0, 0, 0, 0, w, h);
}

__device__ void cuda_vpx_convolve_copy_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	int r;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;

	for (r = h; r > 0; --r)
	{
		memcpy(dst, src, w);
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_scaled_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                        int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                       int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_2d_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                     int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_avg_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                            const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_avg_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                           const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_avg_2d_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ static void cuda_highbd_convolve_horiz(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                  const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h, int bd) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint16_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_highbd_convolve_avg_horiz(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                      const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h, int bd) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint16_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = ROUND_POWER_OF_TWO(dst[x] + cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd), 1);
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_highbd_convolve_vert(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                 const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h, int bd) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint16_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ static void cuda_highbd_convolve_avg_vert(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                     const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h, int bd) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint16_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k)
				sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = ROUND_POWER_OF_TWO( dst[y * dst_stride] + cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd), 1);
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ static void cuda_highbd_convolve(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                            const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                            int h, int bd) {
	uint16_t temp[64 * 135];
	const int intermediate_height =
		(((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

	assert(w <= 64);
	assert(h <= 64);
	assert(y_step_q4 <= 32);
	assert(x_step_q4 <= 32);

	cuda_highbd_convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64, filter, x0_q4, x_step_q4, w,
	                           intermediate_height, bd);
	cuda_highbd_convolve_vert(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride, filter, y0_q4, y_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_horiz_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                  const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                                  int w, int h, int bd) {
	(void)y0_q4;
	(void)y_step_q4;

	cuda_highbd_convolve_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_avg_horiz_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                      const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                                      int w, int h, int bd) {
	(void)y0_q4;
	(void)y_step_q4;

	cuda_highbd_convolve_avg_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_vert_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                 const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                                 int h, int bd) {
	(void)x0_q4;
	(void)x_step_q4;

	cuda_highbd_convolve_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_avg_vert_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                     const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                                     int w, int h, int bd) {
	(void)x0_q4;
	(void)x_step_q4;

	cuda_highbd_convolve_avg_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter, 
											int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd)
{
	cuda_highbd_convolve(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve_avg_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
	const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd) {
	int x, y;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;
	(void)bd;

	for (y = 0; y < h; ++y)
	{
		for (x = 0; x < w; ++x) dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_highbd_convolve8_avg_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                                int h, int bd) {
	// Fixed size intermediate buffer places limits on parameters.
	DECLARE_ALIGNED(16, uint16_t, temp[64 * 64]);
	assert(w <= 64);
	assert(h <= 64);

	cuda_vpx_highbd_convolve8_c(src, src_stride, temp, 64, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd);
	cuda_vpx_highbd_convolve_avg_c(temp, 64, dst, dst_stride, NULL, 0, 0, 0, 0, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve_copy_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                                int h, int bd) {
	int r;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;
	(void)bd;

	for (r = h; r > 0; --r)
	{
		memcpy(dst, src, w * sizeof(uint16_t));
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_high_build_mc_border(const uint8_t* src8, int src_stride,
	uint16_t* dst, int dst_stride, int x, int y,int b_w, int b_h, int w, int h) {
	// Get a pointer to the start of the real data for this row.
	const uint16_t* src = CONVERT_TO_SHORTPTR(src8);
	const uint16_t* ref_row = src - x - y * src_stride;

	if (y >= h) ref_row += (h - 1) * src_stride;
	else if (y > 0)	ref_row += y * src_stride;

	do {
		int right = 0, copy;
		int left = x < 0 ? -x : 0;

		if (left > b_w) left = b_w;

		if (x + b_w > w) right = x + b_w - w;

		if (right > b_w) right = b_w;

		copy = b_w - left - right;

		if (left) cuda_vpx_memset16(dst, ref_row[0], left);
		if (copy) memcpy(dst + left, ref_row + x + left, copy * sizeof(uint16_t));
		if (right) cuda_vpx_memset16(dst + left + copy, ref_row[w - 1], right);

		dst += dst_stride;
		++y;

		if (y > 0 && y < h) ref_row += src_stride;
	} while (--b_h);
}

__device__ static void cuda_highbd_inter_predictor(const uint16_t* src, int src_stride, uint16_t* dst, int dst_stride,
	const int subpel_x, const int subpel_y, const struct scale_factors* sf,
	int w, int h, int ref, const InterpKernel* kernel, int xs, int ys, int bd)
{
	sf->highbd_predict[(subpel_x != 0)][(subpel_y != 0)][ref](src, src_stride, dst, dst_stride, kernel, subpel_x, xs, subpel_y, ys, w, h, bd);
}

__device__ static void cuda_extend_and_predict(const uint8_t* buf_ptr1, int pre_buf_stride, int x0, int y0, int b_w,
			int b_h, int frame_width, int frame_height, int border_offset, uint8_t* const dst, int dst_buf_stride, int subpel_x, int subpel_y,
			const InterpKernel* kernel, const struct scale_factors* sf, int bd, int w, int h, int ref, int xs, int ys) {
#if CONFIG_VP9_HIGHBITDEPTH
	DECLARE_ALIGNED(16, uint16_t, mc_buf_high[80 * 2 * 80 * 2]);
	cuda_high_build_mc_border(buf_ptr1, pre_buf_stride, mc_buf_high, b_w, x0, y0, b_w, b_h, frame_width, frame_height);
	cuda_highbd_inter_predictor(mc_buf_high + border_offset, b_w, CONVERT_TO_SHORTPTR(dst), dst_buf_stride, subpel_x,
		subpel_y, sf, w, h, ref, kernel, xs, ys, bd);
#else
	DECLARE_ALIGNED(16, uint8_t, mc_buf_high[80 * 2 * 80 * 2]);
	cuda_high_build_mc_border(buf_ptr1, pre_buf_stride, mc_buf_high, b_w, x0, y0, b_w, b_h, frame_width, frame_height);
	cuda_highbd_inter_predictor(mc_buf_high + border_offset, b_w, dst, dst_buf_stride, subpel_x,
		subpel_y, sf, w, h, ref, kernel, xs, ys, bd);
#endif
}

__device__ static void cuda_dec_build_inter_predictors(int t_subsampling_x,
		int t_subsampling_y, int mb_to_top_edge, int mb_to_left_edge, int bit_depth, int x, int y, int w, int h,
		const InterpKernel* kernel, const struct scale_factors* sf, int buf_stride, uint8_t* dst_buf, const MV* mv,
		uint8_t* ref_frame, int frame_width, int frame_height, int ref) {
	uint8_t* const dst = dst_buf + buf_stride * y + x;
	MV32 scaled_mv;
	int xs, ys, x0, y0, x0_16, y0_16, subpel_x, subpel_y;
	uint8_t* buf_ptr;
	
	// Co-ordinate of containing block to pixel precision.
	x0 = (-mb_to_left_edge >> (3 + t_subsampling_x)) + x;
	y0 = (-mb_to_top_edge >> (3 + t_subsampling_y)) + y;

	// Co-ordinate of the block to 1/16th pixel precision.
	x0_16 = x0 << SUBPEL_BITS;
	y0_16 = y0 << SUBPEL_BITS;

	scaled_mv.row = mv->row * (1 << (1 - t_subsampling_y));
	scaled_mv.col = mv->col * (1 << (1 - t_subsampling_x));
	xs = ys = 16;
	
	subpel_x = scaled_mv.col & SUBPEL_MASK;
	subpel_y = scaled_mv.row & SUBPEL_MASK;

	// Calculate the top left corner of the best matching block in the
	// reference frame.
	x0 += scaled_mv.col >> SUBPEL_BITS;
	y0 += scaled_mv.row >> SUBPEL_BITS;
	x0_16 += scaled_mv.col;
	y0_16 += scaled_mv.row;

	// Get reference block pointer.
	buf_ptr = ref_frame + y0 * buf_stride + x0;

	// Do border extension if there is motion or the
	// width/height is not a multiple of 8 pixels.
	if (scaled_mv.col || scaled_mv.row || (frame_width & 0x7) ||
		(frame_height & 0x7)) {
		int y1 = ((y0_16 + (h - 1) * ys) >> SUBPEL_BITS) + 1;

		// Get reference block bottom right horizontal coordinate.
		int x1 = ((x0_16 + (w - 1) * xs) >> SUBPEL_BITS) + 1;
		int x_pad = 0, y_pad = 0;

		if (subpel_x || (sf->x_step_q4 != SUBPEL_SHIFTS)) {
			x0 -= VP9_INTERP_EXTEND - 1;
			x1 += VP9_INTERP_EXTEND;
			x_pad = 1;
		}

		if (subpel_y || (sf->y_step_q4 != SUBPEL_SHIFTS)) {
			y0 -= VP9_INTERP_EXTEND - 1;
			y1 += VP9_INTERP_EXTEND;
			y_pad = 1;
		}

		// Skip border extension if block is inside the frame.
		if (x0 < 0 || x0 > frame_width - 1 || x1 < 0 || x1 > frame_width - 1 ||
			y0 < 0 || y0 > frame_height - 1 || y1 < 0 || y1 > frame_height - 1) {
			// Extend the border.
			const uint8_t* const buf_ptr1 = ref_frame + y0 * buf_stride + x0;
			const int b_w = x1 - x0 + 1;
			const int b_h = y1 - y0 + 1;
			const int border_offset = y_pad * 3 * b_w + x_pad * 3;

			cuda_extend_and_predict(buf_ptr1, buf_stride, x0, y0, b_w, b_h, frame_width,
				frame_height, border_offset, dst, buf_stride,
				subpel_x, subpel_y, kernel, sf, bit_depth, w, h, ref, xs, ys);
			return;
		}
	}
	
#if CONFIG_VP9_HIGHBITDEPTH
	cuda_highbd_inter_predictor(CONVERT_TO_SHORTPTR(buf_ptr), buf_stride,
		CONVERT_TO_SHORTPTR(dst), buf_stride, subpel_x,
		subpel_y, sf, w, h, ref, kernel, xs, ys, bit_depth);
#else
	cuda_highbd_inter_predictor(buf_ptr, buf_stride, dst, buf_stride, subpel_x, 
		subpel_y, sf, w, h, ref, kernel, xs, ys, bit_depth);
#endif
}

__device__ static void cuda_dec_build_inter_predictors_4x4(FrameInformation* fi, int subsampling_x,
		int subsampling_y, int is_compound, int interp_filter, MV_REFERENCE_FRAME ref_frame[2], uint8_t* alloc,
		uint8_t* frame_refs[3], struct scale_factors* const sf[3], int16_t my[2], int16_t mx[2], int mi_row, int mi_col, 
		int plane, int x, int y) {
	int size = 0;

	uint8_t* y_buf = (uint8_t*)yv12_align_addr(alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
	uint8_t* u_buf = (uint8_t*)yv12_align_addr(alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
	uint8_t* v_buf = (uint8_t*)yv12_align_addr(alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);

	uint8_t* const buffers[MAX_MB_PLANE] = { y_buf, u_buf, v_buf };
	const int strides[MAX_MB_PLANE] = { fi->y_stride, fi->uv_stride, fi->uv_stride };
	
	const InterpKernel* kernel = cuda_vp9_filter_kernels[interp_filter];
	int ref;

	for (ref = 0; ref < 1 + is_compound; ++ref) {
		int k = ref_frame[ref] - LAST_FRAME;

		uint8_t* ref_alloc = frame_refs[k];
		uint8_t* ref_y_buf = (uint8_t*)yv12_align_addr(ref_alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
		uint8_t* ref_u_buf = (uint8_t*)yv12_align_addr(ref_alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
		uint8_t* ref_v_buf = (uint8_t*)yv12_align_addr(ref_alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
		uint8_t* const ref_buffers[MAX_MB_PLANE] = { ref_y_buf, ref_u_buf, ref_v_buf };
		
		const MV mv = { my[size], mx[size] };
		++size;
		
		uint8_t* dst_buf = buffers[plane] + cuda_scaled_buffer_offset((MI_SIZE * mi_col) >> subsampling_x,
			(MI_SIZE * mi_row) >> subsampling_y, strides[plane], NULL);
		
		const int buf_stride = strides[plane];

		int frame_width;
		int frame_height;
		uint8_t* ref_frame_buf;

		if (plane == 0) {
			frame_width = fi->y_crop_width;
			frame_height = fi->y_crop_height;
		}
		else {
			frame_width = fi->uv_crop_width;
			frame_height = fi->uv_crop_height;
		}
		ref_frame_buf = ref_buffers[plane];

		cuda_dec_build_inter_predictors(subsampling_x, subsampling_y, -(mi_row * MI_SIZE * 8),
			-(mi_col * MI_SIZE * 8), fi->bit_depth, 4 * x, 4 * y, 4, 4, kernel,
			sf[k], buf_stride, dst_buf, &mv, ref_frame_buf, frame_width, frame_height, ref);
	}
}

__global__ static void cuda_inter_4x4(uint8_t* alloc, uint8_t* frame_ref1, uint8_t* frame_ref2, uint8_t* frame_ref3, FrameInformation* fi,
	int16_t* mv, struct scale_factors* const sf1, struct scale_factors* const sf2, struct scale_factors* const sf3, 
	const int super_size, MV_REFERENCE_FRAME* ref_frame, int * block_settings)
{
#if CONFIG_VP9_HIGHBITDEPTH
	uint8_t* frame_refs[3] = { CONVERT_TO_BYTEPTR(frame_ref1), CONVERT_TO_BYTEPTR(frame_ref2), CONVERT_TO_BYTEPTR(frame_ref3) };
#else
	uint8_t* frame_refs[3] = { frame_ref1, frame_ref2, frame_ref3 };
#endif
	
	struct scale_factors* const sf[3] = { sf1, sf2, sf3 };
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int delta = blockDim.x * gridDim.x;
	uint8_t* converted_alloc = alloc;
	
#if CONFIG_VP9_HIGHBITDEPTH
	converted_alloc = CONVERT_TO_BYTEPTR(converted_alloc);
#endif

	MV_REFERENCE_FRAME temp_ref_frame[2];
	int16_t t_mx[2];
	int16_t t_my[2];
	
	if (i < super_size) {
		for(int j = 0; j < 2; ++j)
		{
			temp_ref_frame[j] = ref_frame[2 * i + j];
			t_mx[j] = mv[4 * i + j];
			t_my[j] = mv[4 * i + j + 2];
		}

		//if(i == 0)
		//{
		//	t_mx[0] = 0;
		//	t_my[1] = 0;
		//}
		//else if (i == 1) {
		//	t_mx[0] = 0;
		//	t_my[1] = 1;
		//}
		//else if (i == 2) {
		//	t_mx[0] = 1;
		//	t_my[1] = 0;
		//}
		//else if (i == 3) {
		//	t_mx[0] = 1;
		//	t_my[1] = 1;
		//}
		
		cuda_dec_build_inter_predictors_4x4(fi, block_settings[9 * i + 5], block_settings[9 * i + 6], block_settings[9 * i + 7],
			block_settings[9 * i + 8], temp_ref_frame, converted_alloc, frame_refs, sf, t_my, t_mx, block_settings[9 * i + 3],
			block_settings[9 * i + 4], block_settings[9 * i + 2], block_settings[9 * i], block_settings[9 * i + 1]);
		
	}
}

__host__ void copyFrameInfo(FrameInformation* cd_fi, FrameInformation* fi)
{
	cudaMemcpy(&cd_fi->border, &fi->border, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->size, &fi->size, sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->y_stride, &fi->y_stride, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->uv_stride, &fi->uv_stride, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->uv_border_h, &fi->uv_border_h, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->uv_border_w, &fi->uv_border_w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->yplane_size, &fi->yplane_size, sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->uvplane_size, &fi->uvplane_size, sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->bit_depth, &fi->bit_depth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->vp9_byte_align, &fi->vp9_byte_align, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->y_crop_width, &fi->y_crop_width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->y_crop_height, &fi->y_crop_height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->uv_crop_width, &fi->uv_crop_width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cd_fi->uv_crop_height, &fi->uv_crop_height, sizeof(int), cudaMemcpyHostToDevice);
}

__host__ int createBuffers(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols, 
							int* host_block_settings, MV_REFERENCE_FRAME* host_ref_frame, int16_t* host_mv, FrameInformation* host_fi)
{
	YV12_BUFFER_CONFIG* src = &cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf;
	
	const int uv_border_h = src->border >> src->subsampling_y;
	const int uv_border_w = src->border >> src->subsampling_x;

	const int byte_alignment = cm->byte_alignment;
	const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
	const uint64_t yplane_size = (src->y_height + 2 * src->border) * (uint64_t)src->y_stride + byte_alignment;
	const uint64_t uvplane_size = (src->uv_height + 2 * uv_border_h) * (uint64_t)src->uv_stride + byte_alignment;
	
	host_fi->border = src->border;
	host_fi->size = src->frame_size;
	host_fi->y_stride = src->y_stride;
	host_fi->uv_stride = src->uv_stride;
	host_fi->uv_border_h = uv_border_h;
	host_fi->uv_border_w = uv_border_w;
	host_fi->yplane_size = yplane_size;
	host_fi->uvplane_size = uvplane_size;
	host_fi->bit_depth = cm->bit_depth;
	host_fi->vp9_byte_align = vp9_byte_align;
	host_fi->y_crop_width = src->y_crop_width;
	host_fi->y_crop_height = src->y_crop_height;
	host_fi->uv_crop_width = src->uv_crop_width;
	host_fi->uv_crop_height = src->uv_crop_height;
	
	TileWorkerData* tile_data;
	int super_size = 0;
	int tile_row, mi_col, tile_col, mi_row;
	for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
		TileInfo tile;
		vp9_tile_set_row(&tile, cm, tile_row);
		for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
			for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
				const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
				tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
				vp9_tile_set_col(&tile, cm, col);
				MACROBLOCKD* const xd = &tile_data->xd;
				for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
					for (int i = 0; i < *size_for_mb; ++i) {
						if (is_inter_block(MiBuf->mi[0])) {
							const int bh = 1 << (*MiBuf->bhl - 1);
							const int bw = 1 << (*MiBuf->bwl - 1);
							int plane;
							if (MiBuf->mi[0]->sb_type < BLOCK_8X8) {

							}
							else {
								for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
									int t_subsampling_x = xd->plane[plane].subsampling_x;
									int t_subsampling_y = xd->plane[plane].subsampling_y;

									const int num_4x4_w = (bw << 1) >> t_subsampling_x;
									const int num_4x4_h = (bh << 1) >> t_subsampling_y;

									for (int y = 0; y < num_4x4_h; ++y) {
										for (int x = 0; x < num_4x4_w; ++x) {
											host_ref_frame[2 * super_size + 0] = MiBuf->mi[0]->ref_frame[0];
											host_ref_frame[2 * super_size + 1] = MiBuf->mi[0]->ref_frame[1];
											host_mv[4 * super_size + 0] = MiBuf->mi[0]->mv[0].as_mv.col;
											host_mv[4 * super_size + 2] = MiBuf->mi[0]->mv[0].as_mv.row;
											host_mv[4 * super_size + 1] = MiBuf->mi[0]->mv[1].as_mv.col;
											host_mv[4 * super_size + 3] = MiBuf->mi[0]->mv[1].as_mv.row;
											host_block_settings[9 * super_size + 5] = xd->plane[plane].subsampling_x;
											host_block_settings[9 * super_size + 6] = xd->plane[plane].subsampling_y;
											host_block_settings[9 * super_size + 7] = MiBuf->mi[0]->interp_filter;
											host_block_settings[9 * super_size + 8] = has_second_ref(MiBuf->mi[0]);
											host_block_settings[9 * super_size + 3] = *MiBuf->mi_row;
											host_block_settings[9 * super_size + 4] = *MiBuf->mi_col;
											host_block_settings[9 * super_size + 0] = x;
											host_block_settings[9 * super_size + 1] = y;
											host_block_settings[9 * super_size + 2] = plane;

											++super_size;
										}
									}
								}
							}
						}

						++MiBuf->mi_col;
						++MiBuf->mi;
						++MiBuf->mi_row;
						++MiBuf->bhl;
						++MiBuf->bwl;
					}
					++size_for_mb;
				}
			}
		}
	}

	return super_size;
}

int cuda_inter_prediction(int n, double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
                          VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols)
{
	uint8_t* cd_alloc, * frame_ref1, * frame_ref2, * frame_ref3;
	FrameInformation* cd_fi, * host_fi;
	int * block_settings, * host_block_settings;
	MV_REFERENCE_FRAME* ref_frame, * host_ref_frame;
	int16_t* mv, * host_mv;
	struct scale_factors* sf1, * sf2, * sf3;

	uint8_t* frame_refs[3] = { cm->frame_refs[0].buf->buffer_alloc, cm->frame_refs[1].buf->buffer_alloc, cm->frame_refs[2].buf->buffer_alloc };
	struct scale_factors* const sf[3] = { &cm->frame_refs[0].sf, &cm->frame_refs[1].sf, &cm->frame_refs[2].sf };
	uint8_t* alloc = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.buffer_alloc;
	
	cudaHostAlloc((void**)&host_block_settings, 27 * n / 16 * sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_fi, sizeof(FrameInformation), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_mv, 12 * n / 16 * sizeof(int16_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_ref_frame, 6 * n / 16 * sizeof(MV_REFERENCE_FRAME),cudaHostAllocWriteCombined | cudaHostAllocMapped);

	int super_size = createBuffers(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, host_block_settings, host_ref_frame, host_mv, host_fi);

	cudaMalloc((void**)&cd_alloc, host_fi->size);
	cudaMalloc((void**)&frame_ref1, host_fi->size);
	cudaMalloc((void**)&frame_ref2, host_fi->size);
	cudaMalloc((void**)&frame_ref3, host_fi->size);
	cudaMalloc((void**)&sf1, sizeof(scale_factors));
	cudaMalloc((void**)&sf2, sizeof(scale_factors));
	cudaMalloc((void**)&sf3, sizeof(scale_factors));
	
	clock_t copy_begin = clock();
	//copy
	{
		cudaHostGetDevicePointer((void**)&cd_fi, (void*)host_fi, 0);
		cudaHostGetDevicePointer((void**)&mv, (void*)host_mv, 0);
		cudaHostGetDevicePointer((void**)&block_settings, (void*)host_block_settings, 0);
		cudaHostGetDevicePointer((void**)&ref_frame, (void*)host_ref_frame, 0);
		cudaMemcpy(cd_alloc, alloc, host_fi->size, cudaMemcpyHostToDevice);
		cudaMemcpy(frame_ref1, frame_refs[0], host_fi->size, cudaMemcpyHostToDevice);
		cudaMemcpy(frame_ref2, frame_refs[1], host_fi->size, cudaMemcpyHostToDevice);
		cudaMemcpy(frame_ref3, frame_refs[2], host_fi->size, cudaMemcpyHostToDevice);
		copySF <<<1, 1>>> (sf1, sf[0]->x_scale_fp, sf[0]->y_scale_fp);
		copySF <<<1, 1>>> (sf2, sf[1]->x_scale_fp, sf[1]->y_scale_fp);
		copySF <<<1, 1>>> (sf3, sf[2]->x_scale_fp, sf[2]->y_scale_fp);
	}
	clock_t copy_end = clock();
	*gpu_copy = (double)(copy_end - copy_begin) / CLOCKS_PER_SEC;
	
	setFK <<<1, 1>>> ();
	
	float elapsed = 0;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int blocksPerGrid = MAX(1, (super_size + threadsPerBlock - 1) / threadsPerBlock);
	cuda_inter_4x4 <<<blocksPerGrid, threadsPerBlock>>> (cd_alloc, frame_ref1, frame_ref2, frame_ref3, cd_fi, mv,
															sf1, sf2, sf3, super_size, ref_frame, block_settings);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*gpu_run = ((double)elapsed) / 1000;
	
	cudaError err = cudaGetLastError();
	
	cudaMemcpy(alloc, cd_alloc, host_fi->size, cudaMemcpyDeviceToHost);

	//free
	{
		cudaFreeHost(host_mv);
		cudaFreeHost(host_block_settings);
		cudaFreeHost(host_ref_frame);
		cudaFree(cd_alloc);
		cudaFree(frame_ref1);
		cudaFree(frame_ref2);
		cudaFree(frame_ref3);
		cudaFreeHost(host_fi);
		cudaFree(sf1);
		cudaFree(sf2);
		cudaFree(sf3);
	}
	
	return 0;
}

__global__ void copySF(scale_factors* cuda_sf, int x_scale_fp, int y_scale_fp)
{
	
	cuda_sf->x_scale_fp = x_scale_fp;
	cuda_sf->y_scale_fp = y_scale_fp;
	cuda_sf->x_step_q4 = 16;
	cuda_sf->y_step_q4 = 16;
	cuda_sf->scale_value_x = unscaled_value;
	cuda_sf->scale_value_y = unscaled_value;

	
	// No scaling in either direction.
	cuda_sf->predict[0][0][0] = cuda_vpx_convolve_copy_c;
	cuda_sf->predict[0][0][1] = cuda_vpx_convolve_avg_c;
	cuda_sf->predict[0][1][0] = cuda_vpx_convolve8_vert_c;
	cuda_sf->predict[0][1][1] = cuda_vpx_convolve8_avg_vert_c;
	cuda_sf->predict[1][0][0] = cuda_vpx_convolve8_horiz_c;
	cuda_sf->predict[1][0][1] = cuda_vpx_convolve8_avg_horiz_c;
		
	// 2D subpel motion always gets filtered in both directions
	cuda_sf->predict[1][1][0] = cuda_vpx_convolve8_c;
	cuda_sf->predict[1][1][1] = cuda_vpx_convolve8_avg_c;

	
	// No scaling in either direction.
	cuda_sf->highbd_predict[0][0][0] = cuda_vpx_highbd_convolve_copy_c;
	cuda_sf->highbd_predict[0][0][1] = cuda_vpx_highbd_convolve_avg_c;
	cuda_sf->highbd_predict[0][1][0] = cuda_vpx_highbd_convolve8_vert_c;
	cuda_sf->highbd_predict[0][1][1] = cuda_vpx_highbd_convolve8_avg_vert_c;
	cuda_sf->highbd_predict[1][0][0] = cuda_vpx_highbd_convolve8_horiz_c;
	cuda_sf->highbd_predict[1][0][1] = cuda_vpx_highbd_convolve8_avg_horiz_c;
	
	// 2D subpel motion always gets filtered in both directions.
	cuda_sf->highbd_predict[1][1][0] = cuda_vpx_highbd_convolve8_c;
	cuda_sf->highbd_predict[1][1][1] = cuda_vpx_highbd_convolve8_avg_c;
}

__global__ void setFK()
{
	cuda_vp9_filter_kernels[0] = cuda_sub_pel_filters_8;
	cuda_vp9_filter_kernels[1] = cuda_sub_pel_filters_8lp;
	cuda_vp9_filter_kernels[2] = cuda_sub_pel_filters_8s;
	cuda_vp9_filter_kernels[3] = cuda_bilinear_filters;
	cuda_vp9_filter_kernels[4] = cuda_sub_pel_filters_4;
}