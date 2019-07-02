/*
 * H.26L/H.264/AVC/JVT/14496-10/... decoder
 * Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * H.264 / AVC / MPEG-4 part10 codec.
 * @author Michael Niedermayer <michaelni@gmx.at>
 */

#define UNCHECKED_BITSTREAM_READER 1

#include "libavutil/avassert.h"
#include "libavutil/display.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/stereo3d.h"
#include "libavutil/timer.h"
#include "internal.h"
#include "bytestream.h"
#include "cabac.h"
#include "cabac_functions.h"
#include "error_resilience.h"
#include "avcodec.h"
#include "h264.h"
#include "h264dec.h"
#include "h2645_parse.h"
#include "h264data.h"
#include "h264chroma.h"
#include "h264_mvpred.h"
#include "h264_ps.h"
#include "golomb.h"
#include "hwaccel.h"
#include "mathops.h"
#include "me_cmp.h"
#include "mpegutils.h"
#include "profiles.h"
#include "rectangle.h"
#include "thread.h"
#include "libavutil/motion_vector.h"

const uint16_t ff_h264_mb_sizes[4] = { 256, 384, 512, 768 };

int save_frame_as_jpeg(AVCodecContext *pCodecCtx, AVFrame *pFrame, int FrameNo) {
    AVCodec *jpegCodec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!jpegCodec) {
        return -1;
    }
    AVCodecContext *jpegContext = avcodec_alloc_context3(jpegCodec);
    if (!jpegContext) {
        return -1;
    }

    jpegContext->pix_fmt = AV_PIX_FMT_YUVJ420P;
    jpegContext->height = pFrame->height;
    jpegContext->width = pFrame->width;
    jpegContext->time_base.num = pCodecCtx->time_base.num;
    jpegContext->time_base.den = pCodecCtx->time_base.den;

    if (avcodec_open2(jpegContext, jpegCodec, NULL) < 0) {
        return -1;
    }
    FILE *JPEGFile;
    char JPEGFName[256];

    AVPacket packet = {.data = NULL, .size = 0};
    av_init_packet(&packet);
    int gotFrame;

    if (avcodec_encode_video2(jpegContext, &packet, pFrame, &gotFrame) < 0) {
        return -1;
    }

    sprintf(JPEGFName, "/home/mohamed/FFmpeg/jpeg/%06d.jpg", FrameNo);
    JPEGFile = fopen(JPEGFName, "wb");
    fwrite(packet.data, 1, packet.size, JPEGFile);
    fclose(JPEGFile);

    av_free_packet(&packet);
    avcodec_close(jpegContext);
    return 0;
}


int avpriv_h264_has_num_reorder_frames(AVCodecContext *avctx)
{
    H264Context *h = avctx->priv_data;
    return h && h->ps.sps ? h->ps.sps->num_reorder_frames : 0;
}

static void h264_er_decode_mb(void *opaque, int ref, int mv_dir, int mv_type,
                              int (*mv)[2][4][2],
                              int mb_x, int mb_y, int mb_intra, int mb_skipped)
{
    H264Context *h = opaque;
    H264SliceContext *sl = &h->slice_ctx[0];

    sl->mb_x = mb_x;
    sl->mb_y = mb_y;
    sl->mb_xy = mb_x + mb_y * h->mb_stride;
    memset(sl->non_zero_count_cache, 0, sizeof(sl->non_zero_count_cache));
    av_assert1(ref >= 0);
    /* FIXME: It is possible albeit uncommon that slice references
     * differ between slices. We take the easy approach and ignore
     * it for now. If this turns out to have any relevance in
     * practice then correct remapping should be added. */
    if (ref >= sl->ref_count[0])
        ref = 0;
    if (!sl->ref_list[0][ref].data[0]) {
        av_log(h->avctx, AV_LOG_DEBUG, "Reference not available for error concealing\n");
        ref = 0;
    }
    if ((sl->ref_list[0][ref].reference&3) != 3) {
        av_log(h->avctx, AV_LOG_DEBUG, "Reference invalid\n");
        return;
    }
    fill_rectangle(&h->cur_pic.ref_index[0][4 * sl->mb_xy],
                   2, 2, 2, ref, 1);
    fill_rectangle(&sl->ref_cache[0][scan8[0]], 4, 4, 8, ref, 1);
    fill_rectangle(sl->mv_cache[0][scan8[0]], 4, 4, 8,
                   pack16to32((*mv)[0][0][0], (*mv)[0][0][1]), 4);
    sl->mb_mbaff =
    sl->mb_field_decoding_flag = 0;
    ff_h264_hl_decode_mb(h, &h->slice_ctx[0]);
}

void ff_h264_draw_horiz_band(const H264Context *h, H264SliceContext *sl,
                             int y, int height)
{
    AVCodecContext *avctx = h->avctx;
    const AVFrame   *src  = h->cur_pic.f;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);
    int vshift = desc->log2_chroma_h;
    const int field_pic = h->picture_structure != PICT_FRAME;
    if (field_pic) {
        height <<= 1;
        y      <<= 1;
    }

    height = FFMIN(height, avctx->height - y);

    if (field_pic && h->first_field && !(avctx->slice_flags & SLICE_FLAG_ALLOW_FIELD))
        return;

    if (avctx->draw_horiz_band) {
        int offset[AV_NUM_DATA_POINTERS];
        int i;

        offset[0] = y * src->linesize[0];
        offset[1] =
        offset[2] = (y >> vshift) * src->linesize[1];
        for (i = 3; i < AV_NUM_DATA_POINTERS; i++)
            offset[i] = 0;

        emms_c();

        avctx->draw_horiz_band(avctx, src, offset,
                               y, h->picture_structure, height);
    }
}

void ff_h264_free_tables(H264Context *h)
{
    int i;

    av_freep(&h->intra4x4_pred_mode);
    av_freep(&h->chroma_pred_mode_table);
    av_freep(&h->cbp_table);
    av_freep(&h->mvd_table[0]);
    av_freep(&h->mvd_table[1]);
    av_freep(&h->direct_table);
    av_freep(&h->non_zero_count);
    av_freep(&h->slice_table_base);
    h->slice_table = NULL;
    av_freep(&h->list_counts);

    av_freep(&h->mb2b_xy);
    av_freep(&h->mb2br_xy);

    av_buffer_pool_uninit(&h->qscale_table_pool);
    av_buffer_pool_uninit(&h->mb_type_pool);
    av_buffer_pool_uninit(&h->motion_val_pool);
    av_buffer_pool_uninit(&h->ref_index_pool);

    for (i = 0; i < h->nb_slice_ctx; i++) {
        H264SliceContext *sl = &h->slice_ctx[i];

        av_freep(&sl->dc_val_base);
        av_freep(&sl->er.mb_index2xy);
        av_freep(&sl->er.error_status_table);
        av_freep(&sl->er.er_temp_buffer);

        av_freep(&sl->bipred_scratchpad);
        av_freep(&sl->edge_emu_buffer);
        av_freep(&sl->top_borders[0]);
        av_freep(&sl->top_borders[1]);

        sl->bipred_scratchpad_allocated = 0;
        sl->edge_emu_buffer_allocated   = 0;
        sl->top_borders_allocated[0]    = 0;
        sl->top_borders_allocated[1]    = 0;
    }
}

int ff_h264_alloc_tables(H264Context *h)
{
    const int big_mb_num = h->mb_stride * (h->mb_height + 1);
    const int row_mb_num = 2*h->mb_stride*FFMAX(h->nb_slice_ctx, 1);
    int x, y;

    FF_ALLOCZ_ARRAY_OR_GOTO(h->avctx, h->intra4x4_pred_mode,
                      row_mb_num, 8 * sizeof(uint8_t), fail)
    h->slice_ctx[0].intra4x4_pred_mode = h->intra4x4_pred_mode;

    FF_ALLOCZ_OR_GOTO(h->avctx, h->non_zero_count,
                      big_mb_num * 48 * sizeof(uint8_t), fail)
    FF_ALLOCZ_OR_GOTO(h->avctx, h->slice_table_base,
                      (big_mb_num + h->mb_stride) * sizeof(*h->slice_table_base), fail)
    FF_ALLOCZ_OR_GOTO(h->avctx, h->cbp_table,
                      big_mb_num * sizeof(uint16_t), fail)
    FF_ALLOCZ_OR_GOTO(h->avctx, h->chroma_pred_mode_table,
                      big_mb_num * sizeof(uint8_t), fail)
    FF_ALLOCZ_ARRAY_OR_GOTO(h->avctx, h->mvd_table[0],
                      row_mb_num, 16 * sizeof(uint8_t), fail);
    FF_ALLOCZ_ARRAY_OR_GOTO(h->avctx, h->mvd_table[1],
                      row_mb_num, 16 * sizeof(uint8_t), fail);
    h->slice_ctx[0].mvd_table[0] = h->mvd_table[0];
    h->slice_ctx[0].mvd_table[1] = h->mvd_table[1];

    FF_ALLOCZ_OR_GOTO(h->avctx, h->direct_table,
                      4 * big_mb_num * sizeof(uint8_t), fail);
    FF_ALLOCZ_OR_GOTO(h->avctx, h->list_counts,
                      big_mb_num * sizeof(uint8_t), fail)

    memset(h->slice_table_base, -1,
           (big_mb_num + h->mb_stride) * sizeof(*h->slice_table_base));
    h->slice_table = h->slice_table_base + h->mb_stride * 2 + 1;

    FF_ALLOCZ_OR_GOTO(h->avctx, h->mb2b_xy,
                      big_mb_num * sizeof(uint32_t), fail);
    FF_ALLOCZ_OR_GOTO(h->avctx, h->mb2br_xy,
                      big_mb_num * sizeof(uint32_t), fail);
    for (y = 0; y < h->mb_height; y++)
        for (x = 0; x < h->mb_width; x++) {
            const int mb_xy = x + y * h->mb_stride;
            const int b_xy  = 4 * x + 4 * y * h->b_stride;

            h->mb2b_xy[mb_xy]  = b_xy;
            h->mb2br_xy[mb_xy] = 8 * (FMO ? mb_xy : (mb_xy % (2 * h->mb_stride)));
        }

    return 0;

fail:
    ff_h264_free_tables(h);
    return AVERROR(ENOMEM);
}

/**
 * Init context
 * Allocate buffers which are not shared amongst multiple threads.
 */
int ff_h264_slice_context_init(H264Context *h, H264SliceContext *sl)
{
    ERContext *er = &sl->er;
    int mb_array_size = h->mb_height * h->mb_stride;
    int y_size  = (2 * h->mb_width + 1) * (2 * h->mb_height + 1);
    int c_size  = h->mb_stride * (h->mb_height + 1);
    int yc_size = y_size + 2   * c_size;
    int x, y, i;

    sl->ref_cache[0][scan8[5]  + 1] =
    sl->ref_cache[0][scan8[7]  + 1] =
    sl->ref_cache[0][scan8[13] + 1] =
    sl->ref_cache[1][scan8[5]  + 1] =
    sl->ref_cache[1][scan8[7]  + 1] =
    sl->ref_cache[1][scan8[13] + 1] = PART_NOT_AVAILABLE;

    if (sl != h->slice_ctx) {
        memset(er, 0, sizeof(*er));
    } else
    if (CONFIG_ERROR_RESILIENCE) {

        /* init ER */
        er->avctx          = h->avctx;
        er->decode_mb      = h264_er_decode_mb;
        er->opaque         = h;
        er->quarter_sample = 1;

        er->mb_num      = h->mb_num;
        er->mb_width    = h->mb_width;
        er->mb_height   = h->mb_height;
        er->mb_stride   = h->mb_stride;
        er->b8_stride   = h->mb_width * 2 + 1;

        // error resilience code looks cleaner with this
        FF_ALLOCZ_OR_GOTO(h->avctx, er->mb_index2xy,
                          (h->mb_num + 1) * sizeof(int), fail);

        for (y = 0; y < h->mb_height; y++)
            for (x = 0; x < h->mb_width; x++)
                er->mb_index2xy[x + y * h->mb_width] = x + y * h->mb_stride;

        er->mb_index2xy[h->mb_height * h->mb_width] = (h->mb_height - 1) *
                                                      h->mb_stride + h->mb_width;

        FF_ALLOCZ_OR_GOTO(h->avctx, er->error_status_table,
                          mb_array_size * sizeof(uint8_t), fail);

        FF_ALLOC_OR_GOTO(h->avctx, er->er_temp_buffer,
                         h->mb_height * h->mb_stride * (4*sizeof(int) + 1), fail);

        FF_ALLOCZ_OR_GOTO(h->avctx, sl->dc_val_base,
                          yc_size * sizeof(int16_t), fail);
        er->dc_val[0] = sl->dc_val_base + h->mb_width * 2 + 2;
        er->dc_val[1] = sl->dc_val_base + y_size + h->mb_stride + 1;
        er->dc_val[2] = er->dc_val[1] + c_size;
        for (i = 0; i < yc_size; i++)
            sl->dc_val_base[i] = 1024;
    }

    return 0;

fail:
    return AVERROR(ENOMEM); // ff_h264_free_tables will clean up for us
}

static int h264_init_context(AVCodecContext *avctx, H264Context *h)
{
    int i;

    h->avctx                 = avctx;
    h->cur_chroma_format_idc = -1;

    h->width_from_caller     = avctx->width;
    h->height_from_caller    = avctx->height;

    h->picture_structure     = PICT_FRAME;
    h->workaround_bugs       = avctx->workaround_bugs;
    h->flags                 = avctx->flags;
    h->poc.prev_poc_msb      = 1 << 16;
    h->recovery_frame        = -1;
    h->frame_recovered       = 0;
    h->poc.prev_frame_num    = -1;
    h->sei.frame_packing.arrangement_cancel_flag = -1;
    h->sei.unregistered.x264_build = -1;

    h->next_outputed_poc = INT_MIN;
    for (i = 0; i < MAX_DELAYED_PIC_COUNT; i++)
        h->last_pocs[i] = INT_MIN;

    ff_h264_sei_uninit(&h->sei);

    avctx->chroma_sample_location = AVCHROMA_LOC_LEFT;

    h->nb_slice_ctx = (avctx->active_thread_type & FF_THREAD_SLICE) ? avctx->thread_count : 1;
    h->slice_ctx = av_mallocz_array(h->nb_slice_ctx, sizeof(*h->slice_ctx));
    if (!h->slice_ctx) {
        h->nb_slice_ctx = 0;
        return AVERROR(ENOMEM);
    }

    for (i = 0; i < H264_MAX_PICTURE_COUNT; i++) {
        h->DPB[i].f = av_frame_alloc();
        if (!h->DPB[i].f)
            return AVERROR(ENOMEM);
    }

    h->cur_pic.f = av_frame_alloc();
    if (!h->cur_pic.f)
        return AVERROR(ENOMEM);

    h->last_pic_for_ec.f = av_frame_alloc();
    if (!h->last_pic_for_ec.f)
        return AVERROR(ENOMEM);

    for (i = 0; i < h->nb_slice_ctx; i++)
        h->slice_ctx[i].h264 = h;

    return 0;
}

static av_cold int h264_decode_end(AVCodecContext *avctx)
{
    H264Context *h = avctx->priv_data;
    int i;

    ff_h264_remove_all_refs(h);
    ff_h264_free_tables(h);

    for (i = 0; i < H264_MAX_PICTURE_COUNT; i++) {
        ff_h264_unref_picture(h, &h->DPB[i]);
        av_frame_free(&h->DPB[i].f);
    }
    memset(h->delayed_pic, 0, sizeof(h->delayed_pic));

    h->cur_pic_ptr = NULL;

    av_freep(&h->slice_ctx);
    h->nb_slice_ctx = 0;

    ff_h264_sei_uninit(&h->sei);
    ff_h264_ps_uninit(&h->ps);

    ff_h2645_packet_uninit(&h->pkt);

    ff_h264_unref_picture(h, &h->cur_pic);
    av_frame_free(&h->cur_pic.f);
    ff_h264_unref_picture(h, &h->last_pic_for_ec);
    av_frame_free(&h->last_pic_for_ec.f);

    return 0;
}

static AVOnce h264_vlc_init = AV_ONCE_INIT;

static av_cold int h264_decode_init(AVCodecContext *avctx)
{
    H264Context *h = avctx->priv_data;
    int ret;

    ret = h264_init_context(avctx, h);
    if (ret < 0)
        return ret;

    ret = ff_thread_once(&h264_vlc_init, ff_h264_decode_init_vlc);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "pthread_once has failed.");
        return AVERROR_UNKNOWN;
    }

    if (avctx->ticks_per_frame == 1) {
        if(h->avctx->time_base.den < INT_MAX/2) {
            h->avctx->time_base.den *= 2;
        } else
            h->avctx->time_base.num /= 2;
    }
    avctx->ticks_per_frame = 2;

    if (avctx->extradata_size > 0 && avctx->extradata) {
        ret = ff_h264_decode_extradata(avctx->extradata, avctx->extradata_size,
                                       &h->ps, &h->is_avc, &h->nal_length_size,
                                       avctx->err_recognition, avctx);
        if (ret < 0) {
            h264_decode_end(avctx);
            return ret;
        }
    }

    if (h->ps.sps && h->ps.sps->bitstream_restriction_flag &&
        h->avctx->has_b_frames < h->ps.sps->num_reorder_frames) {
        h->avctx->has_b_frames = h->ps.sps->num_reorder_frames;
    }

    avctx->internal->allocate_progress = 1;

    ff_h264_flush_change(h);

    if (h->enable_er < 0 && (avctx->active_thread_type & FF_THREAD_SLICE))
        h->enable_er = 0;

    if (h->enable_er && (avctx->active_thread_type & FF_THREAD_SLICE)) {
        av_log(avctx, AV_LOG_WARNING,
               "Error resilience with slice threads is enabled. It is unsafe and unsupported and may crash. "
               "Use it at your own risk\n");
    }

    return 0;
}

#if HAVE_THREADS
static int decode_init_thread_copy(AVCodecContext *avctx)
{
    H264Context *h = avctx->priv_data;
    int ret;

    if (!avctx->internal->is_copy)
        return 0;

    memset(h, 0, sizeof(*h));

    ret = h264_init_context(avctx, h);
    if (ret < 0)
        return ret;

    h->context_initialized = 0;

    return 0;
}
#endif

/**
 * instantaneous decoder refresh.
 */
static void idr(H264Context *h)
{
    int i;
    ff_h264_remove_all_refs(h);
    h->poc.prev_frame_num        =
    h->poc.prev_frame_num_offset = 0;
    h->poc.prev_poc_msb          = 1<<16;
    h->poc.prev_poc_lsb          = 0;
    for (i = 0; i < MAX_DELAYED_PIC_COUNT; i++)
        h->last_pocs[i] = INT_MIN;
}

/* forget old pics after a seek */
void ff_h264_flush_change(H264Context *h)
{
    int i, j;

    h->next_outputed_poc = INT_MIN;
    h->prev_interlaced_frame = 1;
    idr(h);

    h->poc.prev_frame_num = -1;
    if (h->cur_pic_ptr) {
        h->cur_pic_ptr->reference = 0;
        for (j=i=0; h->delayed_pic[i]; i++)
            if (h->delayed_pic[i] != h->cur_pic_ptr)
                h->delayed_pic[j++] = h->delayed_pic[i];
        h->delayed_pic[j] = NULL;
    }
    ff_h264_unref_picture(h, &h->last_pic_for_ec);

    h->first_field = 0;
    h->recovery_frame = -1;
    h->frame_recovered = 0;
    h->current_slice = 0;
    h->mmco_reset = 1;
}

/* forget old pics after a seek */
static void flush_dpb(AVCodecContext *avctx)
{
    H264Context *h = avctx->priv_data;
    int i;

    memset(h->delayed_pic, 0, sizeof(h->delayed_pic));

    ff_h264_flush_change(h);
    ff_h264_sei_uninit(&h->sei);

    for (i = 0; i < H264_MAX_PICTURE_COUNT; i++)
        ff_h264_unref_picture(h, &h->DPB[i]);
    h->cur_pic_ptr = NULL;
    ff_h264_unref_picture(h, &h->cur_pic);

    h->mb_y = 0;

    ff_h264_free_tables(h);
    h->context_initialized = 0;
}

static int get_last_needed_nal(H264Context *h)
{
    int nals_needed = 0;
    int first_slice = 0;
    int i, ret;

    for (i = 0; i < h->pkt.nb_nals; i++) {
        H2645NAL *nal = &h->pkt.nals[i];
        GetBitContext gb;

        /* packets can sometimes contain multiple PPS/SPS,
         * e.g. two PAFF field pictures in one packet, or a demuxer
         * which splits NALs strangely if so, when frame threading we
         * can't start the next thread until we've read all of them */
        switch (nal->type) {
        case H264_NAL_SPS:
        case H264_NAL_PPS:
            nals_needed = i;
            break;
        case H264_NAL_DPA:
        case H264_NAL_IDR_SLICE:
        case H264_NAL_SLICE:
            ret = init_get_bits8(&gb, nal->data + 1, nal->size - 1);
            if (ret < 0) {
                av_log(h->avctx, AV_LOG_ERROR, "Invalid zero-sized VCL NAL unit\n");
                if (h->avctx->err_recognition & AV_EF_EXPLODE)
                    return ret;

                break;
            }
            if (!get_ue_golomb_long(&gb) ||  // first_mb_in_slice
                !first_slice ||
                first_slice != nal->type)
                nals_needed = i;
            if (!first_slice)
                first_slice = nal->type;
        }
    }

    return nals_needed;
}

static void debug_green_metadata(const H264SEIGreenMetaData *gm, void *logctx)
{
    av_log(logctx, AV_LOG_DEBUG, "Green Metadata Info SEI message\n");
    av_log(logctx, AV_LOG_DEBUG, "  green_metadata_type: %d\n", gm->green_metadata_type);

    if (gm->green_metadata_type == 0) {
        av_log(logctx, AV_LOG_DEBUG, "  green_metadata_period_type: %d\n", gm->period_type);

        if (gm->period_type == 2)
            av_log(logctx, AV_LOG_DEBUG, "  green_metadata_num_seconds: %d\n", gm->num_seconds);
        else if (gm->period_type == 3)
            av_log(logctx, AV_LOG_DEBUG, "  green_metadata_num_pictures: %d\n", gm->num_pictures);

        av_log(logctx, AV_LOG_DEBUG, "  SEI GREEN Complexity Metrics: %f %f %f %f\n",
               (float)gm->percent_non_zero_macroblocks/255,
               (float)gm->percent_intra_coded_macroblocks/255,
               (float)gm->percent_six_tap_filtering/255,
               (float)gm->percent_alpha_point_deblocking_instance/255);

    } else if (gm->green_metadata_type == 1) {
        av_log(logctx, AV_LOG_DEBUG, "  xsd_metric_type: %d\n", gm->xsd_metric_type);

        if (gm->xsd_metric_type == 0)
            av_log(logctx, AV_LOG_DEBUG, "  xsd_metric_value: %f\n",
                   (float)gm->xsd_metric_value/100);
    }
}

static int decode_nal_units(H264Context *h, const uint8_t *buf, int buf_size)
{
    AVCodecContext *const avctx = h->avctx;
    int nals_needed = 0; ///< number of NALs that need decoding before the next frame thread starts
    int idr_cleared=0;
    int i, ret = 0;

    h->has_slice = 0;
    h->nal_unit_type= 0;

    if (!(avctx->flags2 & AV_CODEC_FLAG2_CHUNKS)) {
        h->current_slice = 0;
        if (!h->first_field) {
            h->cur_pic_ptr = NULL;
            ff_h264_sei_uninit(&h->sei);
        }
    }

    if (h->nal_length_size == 4) {
        if (buf_size > 8 && AV_RB32(buf) == 1 && AV_RB32(buf+5) > (unsigned)buf_size) {
            h->is_avc = 0;
        }else if(buf_size > 3 && AV_RB32(buf) > 1 && AV_RB32(buf) <= (unsigned)buf_size)
            h->is_avc = 1;
    }

    ret = ff_h2645_packet_split(&h->pkt, buf, buf_size, avctx, h->is_avc, h->nal_length_size,
                                avctx->codec_id, avctx->flags2 & AV_CODEC_FLAG2_FAST, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "Error splitting the input into NAL units.\n");
        return ret;
    }

    if (avctx->active_thread_type & FF_THREAD_FRAME)
        nals_needed = get_last_needed_nal(h);
    if (nals_needed < 0)
        return nals_needed;

    for (i = 0; i < h->pkt.nb_nals; i++) {
        H2645NAL *nal = &h->pkt.nals[i];
        int max_slice_ctx, err;

        if (avctx->skip_frame >= AVDISCARD_NONREF &&
            nal->ref_idc == 0 && nal->type != H264_NAL_SEI)
            continue;

        // FIXME these should stop being context-global variables
        h->nal_ref_idc   = nal->ref_idc;
        h->nal_unit_type = nal->type;

        err = 0;
        switch (nal->type) {
        case H264_NAL_IDR_SLICE:
            if ((nal->data[1] & 0xFC) == 0x98) {
                av_log(h->avctx, AV_LOG_ERROR, "Invalid inter IDR frame\n");
                h->next_outputed_poc = INT_MIN;
                ret = -1;
                goto end;
            }
            if(!idr_cleared) {
                idr(h); // FIXME ensure we don't lose some frames if there is reordering
            }
            idr_cleared = 1;
            h->has_recovery_point = 1;
        case H264_NAL_SLICE:
            h->has_slice = 1;

            if ((err = ff_h264_queue_decode_slice(h, nal))) {
                H264SliceContext *sl = h->slice_ctx + h->nb_slice_ctx_queued;
                sl->ref_count[0] = sl->ref_count[1] = 0;
                break;
            }

            if (h->current_slice == 1) {
                if (avctx->active_thread_type & FF_THREAD_FRAME &&
                    i >= nals_needed && !h->setup_finished && h->cur_pic_ptr) {
                    ff_thread_finish_setup(avctx);
                    h->setup_finished = 1;
                }

                if (h->avctx->hwaccel &&
                    (ret = h->avctx->hwaccel->start_frame(h->avctx, buf, buf_size)) < 0)
                    goto end;
            }

            max_slice_ctx = avctx->hwaccel ? 1 : h->nb_slice_ctx;
            if (h->nb_slice_ctx_queued == max_slice_ctx) {
                if (h->avctx->hwaccel) {
                    ret = avctx->hwaccel->decode_slice(avctx, nal->raw_data, nal->raw_size);
                    h->nb_slice_ctx_queued = 0;
                } else
                    ret = ff_h264_execute_decode_slices(h);
                if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))
                    goto end;
            }
            break;
        case H264_NAL_DPA:
        case H264_NAL_DPB:
        case H264_NAL_DPC:
            avpriv_request_sample(avctx, "data partitioning");
            break;
        case H264_NAL_SEI:
            ret = ff_h264_sei_decode(&h->sei, &nal->gb, &h->ps, avctx);
            h->has_recovery_point = h->has_recovery_point || h->sei.recovery_point.recovery_frame_cnt != -1;
            if (avctx->debug & FF_DEBUG_GREEN_MD)
                debug_green_metadata(&h->sei.green_metadata, h->avctx);
            if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))
                goto end;
            break;
        case H264_NAL_SPS: {
            GetBitContext tmp_gb = nal->gb;
            if (avctx->hwaccel && avctx->hwaccel->decode_params) {
                ret = avctx->hwaccel->decode_params(avctx,
                                                    nal->type,
                                                    nal->raw_data,
                                                    nal->raw_size);
                if (ret < 0)
                    goto end;
            }
            if (ff_h264_decode_seq_parameter_set(&tmp_gb, avctx, &h->ps, 0) >= 0)
                break;
            av_log(h->avctx, AV_LOG_DEBUG,
                   "SPS decoding failure, trying again with the complete NAL\n");
            init_get_bits8(&tmp_gb, nal->raw_data + 1, nal->raw_size - 1);
            if (ff_h264_decode_seq_parameter_set(&tmp_gb, avctx, &h->ps, 0) >= 0)
                break;
            ff_h264_decode_seq_parameter_set(&nal->gb, avctx, &h->ps, 1);
            break;
        }
        case H264_NAL_PPS:
            if (avctx->hwaccel && avctx->hwaccel->decode_params) {
                ret = avctx->hwaccel->decode_params(avctx,
                                                    nal->type,
                                                    nal->raw_data,
                                                    nal->raw_size);
                if (ret < 0)
                    goto end;
            }
            ret = ff_h264_decode_picture_parameter_set(&nal->gb, avctx, &h->ps,
                                                       nal->size_bits);
            if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))
                goto end;
            break;
        case H264_NAL_AUD:
        case H264_NAL_END_SEQUENCE:
        case H264_NAL_END_STREAM:
        case H264_NAL_FILLER_DATA:
        case H264_NAL_SPS_EXT:
        case H264_NAL_AUXILIARY_SLICE:
            break;
        default:
            av_log(avctx, AV_LOG_DEBUG, "Unknown NAL code: %d (%d bits)\n",
                   nal->type, nal->size_bits);
        }

        if (err < 0) {
            av_log(h->avctx, AV_LOG_ERROR, "decode_slice_header error\n");
        }
    }

    ret = ff_h264_execute_decode_slices(h);
    if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))
        goto end;

    ret = 0;
end:

#if CONFIG_ERROR_RESILIENCE
    /*
     * FIXME: Error handling code does not seem to support interlaced
     * when slices span multiple rows
     * The ff_er_add_slice calls don't work right for bottom
     * fields; they cause massive erroneous error concealing
     * Error marking covers both fields (top and bottom).
     * This causes a mismatched s->error_count
     * and a bad error table. Further, the error count goes to
     * INT_MAX when called for bottom field, because mb_y is
     * past end by one (callers fault) and resync_mb_y != 0
     * causes problems for the first MB line, too.
     */
    if (!FIELD_PICTURE(h) && h->current_slice &&
        h->ps.sps == (const SPS*)h->ps.sps_list[h->ps.pps->sps_id]->data &&
        h->enable_er) {

        H264SliceContext *sl = h->slice_ctx;
        int use_last_pic = h->last_pic_for_ec.f->buf[0] && !sl->ref_count[0];

        ff_h264_set_erpic(&sl->er.cur_pic, h->cur_pic_ptr);

        if (use_last_pic) {
            ff_h264_set_erpic(&sl->er.last_pic, &h->last_pic_for_ec);
            sl->ref_list[0][0].parent = &h->last_pic_for_ec;
            memcpy(sl->ref_list[0][0].data, h->last_pic_for_ec.f->data, sizeof(sl->ref_list[0][0].data));
            memcpy(sl->ref_list[0][0].linesize, h->last_pic_for_ec.f->linesize, sizeof(sl->ref_list[0][0].linesize));
            sl->ref_list[0][0].reference = h->last_pic_for_ec.reference;
        } else if (sl->ref_count[0]) {
            ff_h264_set_erpic(&sl->er.last_pic, sl->ref_list[0][0].parent);
        } else
            ff_h264_set_erpic(&sl->er.last_pic, NULL);

        if (sl->ref_count[1])
            ff_h264_set_erpic(&sl->er.next_pic, sl->ref_list[1][0].parent);

        sl->er.ref_count = sl->ref_count[0];

        ff_er_frame_end(&sl->er);
        if (use_last_pic)
            memset(&sl->ref_list[0][0], 0, sizeof(sl->ref_list[0][0]));
    }
#endif /* CONFIG_ERROR_RESILIENCE */
    /* clean up */
    if (h->cur_pic_ptr && !h->droppable && h->has_slice) {
        ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,
                                  h->picture_structure == PICT_BOTTOM_FIELD);
    }

    return (ret < 0) ? ret : buf_size;
}

/**
 * Return the number of bytes consumed for building the current frame.
 */
static int get_consumed_bytes(int pos, int buf_size)
{
    if (pos == 0)
        pos = 1;        // avoid infinite loops (I doubt that is needed but...)
    if (pos + 10 > buf_size)
        pos = buf_size; // oops ;)

    return pos;
}

static int output_frame(H264Context *h, AVFrame *dst, H264Picture *srcp)
{
    AVFrame *src = srcp->f;
    int ret;

    ret = av_frame_ref(dst, src);
    if (ret < 0)
        return ret;

    av_dict_set(&dst->metadata, "stereo_mode", ff_h264_sei_stereo_mode(&h->sei.frame_packing), 0);

    if (srcp->sei_recovery_frame_cnt == 0)
        dst->key_frame = 1;

    return 0;
}

static int is_extra(const uint8_t *buf, int buf_size)
{
    int cnt= buf[5]&0x1f;
    const uint8_t *p= buf+6;
    if (!cnt)
        return 0;
    while(cnt--){
        int nalsize= AV_RB16(p) + 2;
        if(nalsize > buf_size - (p-buf) || (p[2] & 0x9F) != 7)
            return 0;
        p += nalsize;
    }
    cnt = *(p++);
    if(!cnt)
        return 0;
    while(cnt--){
        int nalsize= AV_RB16(p) + 2;
        if(nalsize > buf_size - (p-buf) || (p[2] & 0x9F) != 8)
            return 0;
        p += nalsize;
    }
    return 1;
}

static int finalize_frame(H264Context *h, AVFrame *dst, H264Picture *out, int *got_frame)
{
    int ret;

    if (((h->avctx->flags & AV_CODEC_FLAG_OUTPUT_CORRUPT) ||
         (h->avctx->flags2 & AV_CODEC_FLAG2_SHOW_ALL) ||
         out->recovered)) {

        if (!h->avctx->hwaccel &&
            (out->field_poc[0] == INT_MAX ||
             out->field_poc[1] == INT_MAX)
           ) {
            int p;
            AVFrame *f = out->f;
            int field = out->field_poc[0] == INT_MAX;
            uint8_t *dst_data[4];
            int linesizes[4];
            const uint8_t *src_data[4];

            av_log(h->avctx, AV_LOG_DEBUG, "Duplicating field %d to fill missing\n", field);

            for (p = 0; p<4; p++) {
                dst_data[p] = f->data[p] + (field^1)*f->linesize[p];
                src_data[p] = f->data[p] +  field   *f->linesize[p];
                linesizes[p] = 2*f->linesize[p];
            }

            av_image_copy(dst_data, linesizes, src_data, linesizes,
                          f->format, f->width, f->height>>1);
        }

        ret = output_frame(h, dst, out);
        if (ret < 0)
            return ret;

        *got_frame = 1;

        if (CONFIG_MPEGVIDEO) {
            ff_print_debug_info2(h->avctx, dst, NULL,
                                 out->mb_type,
                                 out->qscale_table,
                                 out->motion_val,
                                 NULL,
                                 h->mb_width, h->mb_height, h->mb_stride, 1);
        }
    }

    return 0;
}

static int send_next_delayed_frame(H264Context *h, AVFrame *dst_frame,
                                   int *got_frame, int buf_index)
{
    int ret, i, out_idx;
    H264Picture *out = h->delayed_pic[0];

    h->cur_pic_ptr = NULL;
    h->first_field = 0;

    out_idx = 0;
    for (i = 1;
         h->delayed_pic[i] &&
         !h->delayed_pic[i]->f->key_frame &&
         !h->delayed_pic[i]->mmco_reset;
         i++)
        if (h->delayed_pic[i]->poc < out->poc) {
            out     = h->delayed_pic[i];
            out_idx = i;
        }

    for (i = out_idx; h->delayed_pic[i]; i++)
        h->delayed_pic[i] = h->delayed_pic[i + 1];

    if (out) {
        out->reference &= ~DELAYED_PIC_REF;
        ret = finalize_frame(h, dst_frame, out, got_frame);
        if (ret < 0)
            return ret;
    }

    return buf_index;
}

static int h264_decode_frame(AVCodecContext *avctx, void *data,
                             int *got_frame, AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size       = avpkt->size;
    H264Context *h     = avctx->priv_data;
    AVFrame *pict      = data;
    int buf_index;
    int ret;

    h->flags = avctx->flags;
    h->setup_finished = 0;
    h->nb_slice_ctx_queued = 0;

    ff_h264_unref_picture(h, &h->last_pic_for_ec);

    /* end of stream, output what is still in the buffers */
    if (buf_size == 0)
        return send_next_delayed_frame(h, pict, got_frame, 0);

    if (h->is_avc && av_packet_get_side_data(avpkt, AV_PKT_DATA_NEW_EXTRADATA, NULL)) {
        int side_size;
        uint8_t *side = av_packet_get_side_data(avpkt, AV_PKT_DATA_NEW_EXTRADATA, &side_size);
        if (is_extra(side, side_size))
            ff_h264_decode_extradata(side, side_size,
                                     &h->ps, &h->is_avc, &h->nal_length_size,
                                     avctx->err_recognition, avctx);
    }
    if (h->is_avc && buf_size >= 9 && buf[0]==1 && buf[2]==0 && (buf[4]&0xFC)==0xFC) {
        if (is_extra(buf, buf_size))
            return ff_h264_decode_extradata(buf, buf_size,
                                            &h->ps, &h->is_avc, &h->nal_length_size,
                                            avctx->err_recognition, avctx);
    }

    buf_index = decode_nal_units(h, buf, buf_size);
    if (buf_index < 0)
        return AVERROR_INVALIDDATA;

    if (!h->cur_pic_ptr && h->nal_unit_type == H264_NAL_END_SEQUENCE) {
        av_assert0(buf_index <= buf_size);
        return send_next_delayed_frame(h, pict, got_frame, buf_index);
    }

    if (!(avctx->flags2 & AV_CODEC_FLAG2_CHUNKS) && (!h->cur_pic_ptr || !h->has_slice)) {
        if (avctx->skip_frame >= AVDISCARD_NONREF ||
            buf_size >= 4 && !memcmp("Q264", buf, 4))
            return buf_size;
        av_log(avctx, AV_LOG_ERROR, "no frame!\n");
        return AVERROR_INVALIDDATA;
    }

    if (!(avctx->flags2 & AV_CODEC_FLAG2_CHUNKS) ||
        (h->mb_y >= h->mb_height && h->mb_height)) {
        if ((ret = ff_h264_field_end(h, &h->slice_ctx[0], 0)) < 0)
            return ret;

        /* Wait for second field. */
        if (h->next_output_pic) {
            // printf("avctx->gop_size %d \n",avctx->gop_size );
            save_frame_as_jpeg(avctx, h->next_output_pic->f, 2 * h->next_output_pic->f->coded_picture_number);
            //ret = finalize_frame(h, pict, h->next_output_pic, got_frame);

            // switch(h->slice_ctx[0].slice_type_nos) {
            //     case 1:
            //     printf("slice_type = AV_PICTURE_TYPE_I\n");
            //     break;
            //     case 2:
            //     printf("slice_type = AV_PICTURE_TYPE_P\n");
            //     break;
            //     case 3:
            //     printf("slice_type = AV_PICTURE_TYPE_B\n");
            //     break;
            //     default:
            //     printf("slice_type != I or P or B\n");
            // }
            if(h->slice_ctx[0].slice_type_nos == AV_PICTURE_TYPE_I){
                h->following_interpolated_frame = NULL;
            }

            h->next_output_pic->f->interpolated_frame = NULL;
            AVFrame* next_interpolated_frame = NULL;
            if(h->slice_ctx[0].slice_type_nos == AV_PICTURE_TYPE_P) {

                H264Picture * pic = h->next_output_pic;

                h->next_output_pic->f->interpolated_frame = av_frame_alloc();
                next_interpolated_frame = av_frame_alloc();

                if (!h->next_output_pic->f->interpolated_frame)
                    printf("Could not allocate interpolated_frame\n");
                if (!next_interpolated_frame)
                    printf("Could not allocate following_interpolated_frame \n");

                AVFrame* interpolated_frame = h->next_output_pic->f->interpolated_frame;
                interpolated_frame->format = avctx->pix_fmt;
                interpolated_frame->width  = avctx->width;
                interpolated_frame->height = avctx->height;
                // AVFrame* next_interpolated_frame = h->following_interpolated_frame;
                next_interpolated_frame->format = avctx->pix_fmt;
                next_interpolated_frame->width  = avctx->width;
                next_interpolated_frame->height = avctx->height;

                ret = av_frame_get_buffer(interpolated_frame, 64);
                if (ret < 0)
                    printf("Could not allocate the video frame data\n");
                
                ret = av_frame_get_buffer(next_interpolated_frame, 64);
                if (ret < 0)
                    printf("Could not allocate the next_interpolated_frame data\n");

                // make sure the frame data is writable
                ret = av_frame_make_writable(interpolated_frame);
                if (ret < 0)
                    printf("Could not make frame data writable\n");

                // make sure the frame data is writable
                ret = av_frame_make_writable(next_interpolated_frame);
                if (ret < 0)
                    printf("Could not make frame data writable\n");

                int DPB_map[128];    // maps frame_num to its index in the DPB array
                memset(DPB_map, 0, 128 * sizeof(int)); 
                int DPB_count = 0;
                int max_DPB = -1;
                for(int i = H264_MAX_PICTURE_COUNT-1; i >= 0; i--){
                    if (h->DPB[i].f->buf[0]) {
                        max_DPB = i;
                        break;
                    }
                }

/****AVERAGE*********AVERAGE********AVERAGE********AVERAGE*********************************/
/****AVERAGE*********AVERAGE********AVERAGE********AVERAGE*********************************/
                // loop on the DPB array to fill the DPB map
                for (int i = 0; i < H264_MAX_PICTURE_COUNT; i++) {
                    // printf("ZZZ out\n");
                    if (h->DPB[i].f->buf[0]) {
                        // printf("ZZZ in\n");
                        DPB_count++;
                        // printf("DPB frame num %d\n", h->DPB[i].frame_num);
                        DPB_map[h->DPB[i].frame_num] = i;

                        if (h->DPB[i].frame_num + 1 == pic->frame_num ||
                            h->DPB[i].frame_num == max_DPB && pic->frame_num == 0) {

                            for (int y = 0; y < avctx->height; y++) {
                                for (int x = 0; x < avctx->width; x++) {
                                    // Y
                                    interpolated_frame->data[0][y * interpolated_frame->linesize[0] + x] = 
                                    (h->DPB[i].f->data[0][y * h->DPB[i].f->linesize[0] + x] / 2)
                                    + (pic->f->data[0][y * pic->f->linesize[0] + x] / 2);

                                    if(y < avctx->height/2 && x < avctx->width/2){
                                        //Cb
                                        interpolated_frame->data[1][y * interpolated_frame->linesize[1] + x] = 
                                        (h->DPB[i].f->data[1][y * h->DPB[i].f->linesize[1] + x] / 2)
                                        + (pic->f->data[1][y * pic->f->linesize[1] + x] / 2);

                                        //Cr
                                        interpolated_frame->data[2][y * interpolated_frame->linesize[2] + x] = 
                                        (h->DPB[i].f->data[2][y * h->DPB[i].f->linesize[2] + x] / 2)
                                        + (pic->f->data[2][y * pic->f->linesize[2] + x] / 2);
                                    }
                                }
                            }
                        }
                    }
                }
/****END*********END********END********END*************************************************/



                int l0ref_count = pic->ref_count[0][0];
                int l1ref_count = pic->ref_count[0][1];
                for(int i = 0; i < l0ref_count; ++i) {
                    // printf("l0ref%d_poc = %d\n", i+1, pic->ref_poc[0][0][i]);
                    // printf("l0ref%d_frame_num = %d\n", i+1, (pic->ref_poc[0][0][i] - (h->slice_ctx[0].ref_list[0][i].reference & 3)) / 4);
                    if(!h->DPB[DPB_map[(pic->ref_poc[0][0][i] - (h->slice_ctx[0].ref_list[0][i].reference & 3)) / 4]].f->buf[0]) {
                        printf("Frame is not in DPB\n" );
                    }
                }
                for(int i = 0; i < l1ref_count; ++i){
                    // printf("l1ref_poc%d = %d\n", i+1, pic->ref_poc[0][1][i]);
                    // printf("l1ref%d_frame_num = %d\n", i+1, (pic->ref_poc[0][1][i] - (h->slice_ctx[0].ref_list[1][i].reference & 3)) / 4);
                    if(!h->DPB[DPB_map[(pic->ref_poc[0][1][i] - (h->slice_ctx[0].ref_list[1][i].reference & 3)) / 4]].f->buf[0]) {
                        printf("Frame is not in DPB\n" );
                    }
                }
                
                // TODO: make this dynamic in size
                int vis[720 /16 * 2][1280 /16 * 2];
                for(int i = 0; i < h->mb_height*2; i++)
                    memset(vis[i], 0, h->mb_width*2 * sizeof(int));
                // TODO: make this dynamic in size
                for(int i = 0; i < h->mb_height*2; i++)
                    memset(h->next_vis[i], 0, h->mb_width*2 * sizeof(int));
                // TODO: make this dynamic in size
                int mvss[720 / 16 *2][1280 / 16 *2][2];
                for(int i = 0; i < h->mb_height*2; i++)
                    for(int j = 0; j < h->mb_width*2; j++)
                        memset(mvss[i][j], 0, 2 * sizeof(int));

                const int shift = 2;
                const int scale = 1 << shift;
                const int mv_sample_log2 = 2;
                const int mv_stride      = h->mb_width << mv_sample_log2;
                int mb_y, mb_x, mbcount = 0;


                // loop on the picture macroblocks to get motion vectors
                // and construct interpolated frame from [mv, ref_index, ref_pocs, DPB]
                for (mb_y = 0; mb_y < h->mb_height; mb_y++) {
                    for (mb_x = 0; mb_x < h->mb_width; mb_x++) {
                        int mb_xy = mb_x + mb_y * h->mb_stride;
                        int i, direction, mb_type = pic->mb_type[mb_x + mb_y * h->mb_stride];
                        
                        for (direction = 0; direction < 2; direction++) {
                            if (!USES_LIST(mb_type, direction))
                                continue;
                            int ref_index = pic->ref_index[direction][mb_xy];
                            int prev = (pic->ref_poc[0][direction][ref_index] - (h->slice_ctx[0].ref_list[direction][ref_index].reference & 3)) / 4;
                            int curr = pic->frame_num;
                            int range = (curr - prev + max_DPB + 1)%(max_DPB+1);
                            double mv_ratio = 1-((range-0.5)/range);
                            // printf("curr:%d, prev:%d , range:%d, ratio:%f\n",curr, prev, range, mv_ratio );
                            if (IS_8X8(mb_type)) {
                                for (i = 0; i < 4; i++) {
                                    int xy = (mb_x * 2 + (i & 1) +
                                              (mb_y * 2 + (i >> 1)) * mv_stride) << (mv_sample_log2 - 1);
                                    int mx = pic->motion_val[direction][xy][0];
                                    int my = pic->motion_val[direction][xy][1];
                                    int ref_index = pic->ref_index[direction][mb_xy];
                                    int ref_frame_num = (pic->ref_poc[0][direction][ref_index] - (h->slice_ctx[0].ref_list[direction][ref_index].reference & 3)) / 4;
                                    int ref_frame_index = DPB_map[ref_frame_num];
                                    int start_x = (mb_x * 16 + 8 * (i & 1))  ;
                                    int start_y = (mb_y * 16 + 8 * (i >> 1)) ;
                                    int dst_x   = (mb_x * 16 + 8 * (i & 1))  + (mx*mv_ratio)/scale;
                                    int dst_y   = (mb_y * 16 + 8 * (i >> 1)) + (my*mv_ratio)/scale;
                                    if(start_x > (h->mb_width * 16) - 8 || start_x < 0) {
                                        continue;
                                        start_x = mb_x * 16 + 8 * (i & 1);
                                    }
                                    if(start_y > (h->mb_height * 16) - 8 || start_y < 0) {
                                        continue;
                                        start_y = mb_y * 16 + 8 * (i >> 1);
                                    }
                                    if(dst_x > (h->mb_width * 16) - 8 || dst_x < 0) {
                                        continue;
                                        dst_x = mb_x * 16 + 8 * (i & 1);
                                    }
                                    if(dst_y > (h->mb_height * 16) - 8 || dst_y < 0) {
                                        continue;
                                        dst_y = mb_y * 16 + 8 * (i >> 1);
                                    }
                                    int next_start_x = (mb_x * 16 + 8 * (i & 1))  ;
                                    int next_start_y = (mb_y * 16 + 8 * (i >> 1)) ;
                                    int next_dst_x   = (mb_x * 16 + 8 * (i & 1))  - (mx*mv_ratio)/scale;
                                    int next_dst_y   = (mb_y * 16 + 8 * (i >> 1)) - (my*mv_ratio)/scale;
                                    if(next_start_x > (h->mb_width * 16) - 8 || next_start_x < 0) {
                                        continue;
                                        next_start_x = mb_x * 16 + 8 * (i & 1);
                                    }
                                    if(next_start_y > (h->mb_height * 16) - 8 || next_start_y < 0) {
                                        continue;
                                        next_start_y = mb_y * 16 + 8 * (i >> 1);
                                    }
                                    if(next_dst_x > (h->mb_width * 16) - 8 || next_dst_x < 0) {
                                        continue;
                                        next_dst_x = mb_x * 16 + 8 * (i & 1);
                                    }
                                    if(next_dst_y > (h->mb_height * 16) - 8 || next_dst_y < 0) {
                                        continue;
                                        next_dst_y = mb_y * 16 + 8 * (i >> 1);
                                    }


                                    vis[dst_y/8][dst_x/8] = 1;
                                    mvss[dst_y/8][dst_x/8][0] = mx;
                                    mvss[dst_y/8][dst_x/8][1] = my;

                                    h->next_vis[next_dst_y/8][next_dst_x/8] = 1;

                                    int y, x, y_ref, x_ref;
                                    int next_y, next_x, next_y_ref, next_x_ref;

                                    for (y = start_y, y_ref = dst_y,
                                        next_y = next_start_y, next_y_ref = next_dst_y
                                        ; y < start_y + 8;
                                         y++, y_ref++,
                                         next_y++, next_y_ref++) {
                                        for (x = start_x, x_ref = dst_x,
                                        next_x = next_start_x, next_x_ref = next_dst_x;
                                         x < start_x + 8;
                                          x++, x_ref++,
                                          next_x++, next_x_ref++) {
                                            // Y
                                            interpolated_frame->data[0][y_ref * interpolated_frame->linesize[0] + x_ref]
                                            = pic->f->data[0][y * pic->f->linesize[0] + x];

                                            // next Y
                                            next_interpolated_frame->data[0][next_y_ref * next_interpolated_frame->linesize[0] + next_x_ref]
                                            = pic->f->data[0][next_y * pic->f->linesize[0] + next_x];
                                        }
                                    }
                                    for (y = start_y/2, y_ref = dst_y/2,
                                    next_y = next_start_y/2, next_y_ref = next_dst_y/2; 
                                    y < start_y/2 + (8/2);
                                     y++, y_ref++,
                                     next_y++, next_y_ref++
                                     ) {
                                        for (x = start_x/2, x_ref = dst_x/2,
                                        next_x = next_start_x/2, next_x_ref = next_dst_x/2; 
                                        x < start_x/2 + (8/2);
                                         x++, x_ref++,
                                         next_x++, next_x_ref++) {
                                            //Cb
                                            interpolated_frame->data[1][y_ref * interpolated_frame->linesize[1] + x_ref]
                                            = pic->f->data[1][y * pic->f->linesize[1] + x];
                                            
                                            // next_ Cb
                                            next_interpolated_frame->data[1][next_y_ref * next_interpolated_frame->linesize[1] + next_x_ref]
                                            = pic->f->data[1][next_y * pic->f->linesize[1] + next_x];


                                            //Cr
                                            interpolated_frame->data[2][y_ref * interpolated_frame->linesize[2] + x_ref]
                                            = pic->f->data[2][y * pic->f->linesize[2] + x];

                                            // next_ Cr
                                            next_interpolated_frame->data[2][next_y_ref * interpolated_frame->linesize[2] + next_x_ref]
                                            = pic->f->data[2][next_y * pic->f->linesize[2] + next_x];
                                        }
                                    }
                                }
                            } else if (IS_16X8(mb_type)) {
                                for (i = 0; i < 2; i++) {
                                    //int sx = mb_x * 16 + 8;
                                    //int sy = mb_y * 16 + 4 + 8 * i;
                                    int xy = (mb_x * 2 + (mb_y * 2 + i) * mv_stride) << (mv_sample_log2 - 1);
                                    int mx = pic->motion_val[direction][xy][0];
                                    int my = pic->motion_val[direction][xy][1];
                                    // printf("2 mx %d, my %d\n", mx, my);
                                    int ref_index = pic->ref_index[direction][mb_xy];
                                    // printf("2 ref_index %d\n", ref_index);
                                    
                                    int ref_frame_num = (pic->ref_poc[0][direction][ref_index] - (h->slice_ctx[0].ref_list[direction][ref_index].reference & 3)) / 4;
                                    int ref_frame_index = DPB_map[ref_frame_num];
                                    int start_x = (mb_x * 16)         ;
                                    int start_y = (mb_y * 16 + 8 * i) ;
                                    int dst_x   = (mb_x * 16)         + (mx*mv_ratio)/scale;
                                    int dst_y   = (mb_y * 16 + 8 * i) + (my*mv_ratio)/scale;
                                    if(start_x > (h->mb_width * 16) - 16 || start_x < 0) {
                                        continue;
                                        start_x = mb_x * 16;
                                    }
                                    if(start_y > (h->mb_height * 16) - 8 || start_y < 0) {
                                        continue;
                                        start_y = mb_y * 16 + 8 * i;
                                    }
                                    if(dst_x > (h->mb_width * 16) - 16 || dst_x < 0) {
                                        continue;
                                        dst_x = mb_x * 16;
                                    }
                                    if(dst_y > (h->mb_height * 16) - 8 || dst_y < 0) {
                                        continue;
                                        dst_y = mb_y * 16 + 8 * i;
                                    }
                                    // printf("dst_x %d, dst_y %d\n", dst_x, dst_y);
                                    // printf("start_x %d, start_y %d\n", start_x, start_y);

                                    int next_start_x = (mb_x * 16)         ;
                                    int next_start_y = (mb_y * 16 + 8 * i) ;
                                    int next_dst_x   = (mb_x * 16)         - (mx*mv_ratio)/scale;
                                    int next_dst_y   = (mb_y * 16 + 8 * i) - (my*mv_ratio)/scale;
                                    if(next_start_x > (h->mb_width * 16) - 16 || next_start_x < 0) {
                                        continue;
                                        next_start_x = mb_x * 16;
                                    }
                                    if(next_start_y > (h->mb_height * 16) - 8 || next_start_y < 0) {
                                        continue;
                                        next_start_y = mb_y * 16 + 8 * i;
                                    }
                                    if(next_dst_x > (h->mb_width * 16) - 16 || next_dst_x < 0) {
                                        continue;
                                        next_dst_x = mb_x * 16;
                                    }
                                    if(next_dst_y > (h->mb_height * 16) - 8 || next_dst_y < 0) {
                                        continue;
                                        next_dst_y = mb_y * 16 + 8 * i;
                                    }



                                    vis[dst_y/8][dst_x/8] = 1;
                                    vis[dst_y/8][dst_x/8 + 1] = 1;
                                    mvss[dst_y/8][dst_x/8][0] = mx;
                                    mvss[dst_y/8][dst_x/8][1] = my;
                                    mvss[dst_y/8][dst_x/8 + 1][0] = mx;
                                    mvss[dst_y/8][dst_x/8 + 1][1] = my;

                                    h->next_vis[next_dst_y/8][next_dst_x/8] = 1;
                                    h->next_vis[next_dst_y/8][next_dst_x/8 + 1] = 1;


                                    int y, x, y_ref, x_ref;
                                    int next_y, next_x, next_y_ref, next_x_ref;
                                    for (y = start_y, y_ref = dst_y,
                                        next_y = next_start_y, next_y_ref = next_dst_y
                                        ; y < start_y + 8;
                                         y++, y_ref++,
                                         next_y++, next_y_ref++) {
                                        for (x = start_x, x_ref = dst_x,
                                        next_x = next_start_x, next_x_ref = next_dst_x;
                                         x < start_x + 16;
                                          x++, x_ref++,
                                          next_x++, next_x_ref++) {
                                            // Y
                                            interpolated_frame->data[0][y_ref * interpolated_frame->linesize[0] + x_ref]
                                            = pic->f->data[0][y * pic->f->linesize[0] + x];

                                            // next Y
                                            next_interpolated_frame->data[0][next_y_ref * next_interpolated_frame->linesize[0] + next_x_ref]
                                            = pic->f->data[0][next_y * pic->f->linesize[0] + next_x];
                                        }
                                    }
                                    for (y = start_y/2, y_ref = dst_y/2,
                                    next_y = next_start_y/2, next_y_ref = next_dst_y/2; 
                                    y < start_y/2 + (8/2);
                                     y++, y_ref++,
                                     next_y++, next_y_ref++
                                     ) {
                                        for (x = start_x/2, x_ref = dst_x/2,
                                        next_x = next_start_x/2, next_x_ref = next_dst_x/2; 
                                        x < start_x/2 + (16/2);
                                         x++, x_ref++,
                                         next_x++, next_x_ref++) {
                                            //Cb
                                            interpolated_frame->data[1][y_ref * interpolated_frame->linesize[1] + x_ref]
                                            = pic->f->data[1][y * pic->f->linesize[1] + x];
                                            
                                            // next_ Cb
                                            next_interpolated_frame->data[1][next_y_ref * next_interpolated_frame->linesize[1] + next_x_ref]
                                            = pic->f->data[1][next_y * pic->f->linesize[1] + next_x];


                                            //Cr
                                            interpolated_frame->data[2][y_ref * interpolated_frame->linesize[2] + x_ref]
                                            = pic->f->data[2][y * pic->f->linesize[2] + x];

                                            // next_ Cr
                                            next_interpolated_frame->data[2][next_y_ref * interpolated_frame->linesize[2] + next_x_ref]
                                            = pic->f->data[2][next_y * pic->f->linesize[2] + next_x];
                                        }
                                    }

                                    mbcount++;
                                    //if (IS_INTERLACED(mb_type))
                                    //    my *= 2;

                                    //mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx, my, scale, direction);
                                }
                            } else if (IS_8X16(mb_type)) {
                                for (i = 0; i < 2; i++) {
                                    //int sx = mb_x * 16 + 4 + 8 * i;
                                    //int sy = mb_y * 16 + 8;
                                    int xy = (mb_x * 2 + i + mb_y * 2 * mv_stride) << (mv_sample_log2 - 1);
                                    int mx = pic->motion_val[direction][xy][0];
                                    int my = pic->motion_val[direction][xy][1];
                                    // printf("2 mx %d, my %d\n", mx, my);
                                    int ref_index = pic->ref_index[direction][mb_xy];
                                    // printf("2 ref_index %d\n", ref_index);
                                    
                                    int ref_frame_num = (pic->ref_poc[0][direction][ref_index] - (h->slice_ctx[0].ref_list[direction][ref_index].reference & 3)) / 4;
                                    int ref_frame_index = DPB_map[ref_frame_num];
                                    int start_x = (mb_x * 16 + 8 * i) ;
                                    int start_y = (mb_y * 16)         ;
                                    int dst_x   = (mb_x * 16 + 8 * i) + (mx*mv_ratio)/scale;
                                    int dst_y   = (mb_y * 16)         + (my*mv_ratio)/scale;
                                    if(start_x > (h->mb_width * 16) - 8 || start_x < 0) {
                                        continue;
                                        start_x = mb_x * 16 + 8 * i;
                                    }
                                    if(start_y > (h->mb_height * 16) - 16 || start_y < 0) {
                                        continue;
                                        start_y = mb_y * 16;
                                    }
                                    if(dst_x > (h->mb_width * 16) - 8 || dst_x < 0) {
                                        continue;
                                        dst_x = mb_x * 16 + 8 * i;
                                    }
                                    if(dst_y > (h->mb_height * 16) - 16 || dst_y < 0) {
                                        continue;
                                        dst_y = mb_y * 16;
                                    }
                                    // printf("dst_x %d, dst_y %d\n", dst_x, dst_y);
                                    // printf("start_x %d, start_y %d\n", start_x, start_y);


                                    int next_start_x = (mb_x * 16 + 8 * i) ;
                                    int next_start_y = (mb_y * 16)         ;
                                    int next_dst_x   = (mb_x * 16 + 8 * i) - (mx*mv_ratio)/scale;
                                    int next_dst_y   = (mb_y * 16)         - (my*mv_ratio)/scale;
                                    if(next_start_x > (h->mb_width * 16) - 8 || next_start_x < 0) {
                                        continue;
                                        next_start_x = mb_x * 16 + 8 * i;
                                    }
                                    if(next_start_y > (h->mb_height * 16) - 16 || next_start_y < 0) {
                                        continue;
                                        next_start_y = mb_y * 16;
                                    }
                                    if(next_dst_x > (h->mb_width * 16) - 8 || next_dst_x < 0) {
                                        continue;
                                        next_dst_x = mb_x * 16 + 8 * i;
                                    }
                                    if(next_dst_y > (h->mb_height * 16) - 16 || next_dst_y < 0) {
                                        continue;
                                        next_dst_y = mb_y * 16;
                                    }





                                    vis[dst_y/8][dst_x/8] = 1;
                                    vis[dst_y/8 + 1][dst_x/8] = 1;
                                    mvss[dst_y/8][dst_x/8][0] = mx;
                                    mvss[dst_y/8][dst_x/8][1] = my;
                                    mvss[dst_y/8 + 1][dst_x/8][0] = mx;
                                    mvss[dst_y/8 + 1][dst_x/8][1] = my;

                                    h->next_vis[next_dst_y/8][next_dst_x/8] = 1;
                                    h->next_vis[next_dst_y/8 + 1][next_dst_x/8] = 1;

                                    int y, x, y_ref, x_ref;
                                    int next_y, next_x, next_y_ref, next_x_ref;
                                    for (y = start_y, y_ref = dst_y,
                                        next_y = next_start_y, next_y_ref = next_dst_y
                                        ; y < start_y + 16;
                                         y++, y_ref++,
                                         next_y++, next_y_ref++) {
                                        for (x = start_x, x_ref = dst_x,
                                        next_x = next_start_x, next_x_ref = next_dst_x;
                                         x < start_x + 8;
                                          x++, x_ref++,
                                          next_x++, next_x_ref++) {
                                            // Y
                                            interpolated_frame->data[0][y_ref * interpolated_frame->linesize[0] + x_ref]
                                            = pic->f->data[0][y * pic->f->linesize[0] + x];

                                            // next Y
                                            next_interpolated_frame->data[0][next_y_ref * next_interpolated_frame->linesize[0] + next_x_ref]
                                            = pic->f->data[0][next_y * pic->f->linesize[0] + next_x];
                                        }
                                    }
                                    for (y = start_y/2, y_ref = dst_y/2,
                                    next_y = next_start_y/2, next_y_ref = next_dst_y/2; 
                                    y < start_y/2 + (16/2);
                                     y++, y_ref++,
                                     next_y++, next_y_ref++
                                     ) {
                                        for (x = start_x/2, x_ref = dst_x/2,
                                        next_x = next_start_x/2, next_x_ref = next_dst_x/2; 
                                        x < start_x/2 + (8/2);
                                         x++, x_ref++,
                                         next_x++, next_x_ref++) {
                                            //Cb
                                            interpolated_frame->data[1][y_ref * interpolated_frame->linesize[1] + x_ref]
                                            = pic->f->data[1][y * pic->f->linesize[1] + x];
                                            
                                            // next_ Cb
                                            next_interpolated_frame->data[1][next_y_ref * next_interpolated_frame->linesize[1] + next_x_ref]
                                            = pic->f->data[1][next_y * pic->f->linesize[1] + next_x];


                                            //Cr
                                            interpolated_frame->data[2][y_ref * interpolated_frame->linesize[2] + x_ref]
                                            = pic->f->data[2][y * pic->f->linesize[2] + x];

                                            // next_ Cr
                                            next_interpolated_frame->data[2][next_y_ref * interpolated_frame->linesize[2] + next_x_ref]
                                            = pic->f->data[2][next_y * pic->f->linesize[2] + next_x];
                                        }
                                    }

                                    mbcount++;
                                    //if (IS_INTERLACED(mb_type))
                                    //    my *= 2;

                                    //mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx, my, scale, direction);
                                }
                            } else {
                                //int sx = mb_x * 16 + 8;
                                //int sy = mb_y * 16 + 8;
                                int xy = (mb_x + mb_y * mv_stride) << mv_sample_log2;
                                int mx = pic->motion_val[direction][xy][0];
                                int my = pic->motion_val[direction][xy][1];
                                // printf("1 mx %d, my %d\n", mx, my);
                                int ref_index = pic->ref_index[direction][mb_xy];
                                // printf("1 ref_index %d\n", ref_index);
                                    
                                int ref_frame_num = (pic->ref_poc[0][direction][ref_index] - (h->slice_ctx[0].ref_list[direction][ref_index].reference & 3)) / 4;
                                int ref_frame_index = DPB_map[ref_frame_num];
                                int start_x = (mb_x * 16) ;
                                int start_y = (mb_y * 16) ;
                                int dst_x   = (mb_x * 16) + (mx*mv_ratio)/scale;
                                int dst_y   = (mb_y * 16) + (my*mv_ratio)/scale;
                                if(start_x > (h->mb_width * 16) - 16 || start_x < 0) {
                                    continue;
                                    start_x = mb_x * 16;
                                }
                                if(start_y > (h->mb_height * 16) - 16 || start_y < 0) {
                                    continue;
                                    start_y = mb_y * 16;
                                }
                                if(dst_x > (h->mb_width * 16) - 16 || dst_x < 0) {
                                    continue;
                                    dst_x = mb_x * 16;
                                }
                                if(dst_y > (h->mb_height * 16) - 16 || dst_y < 0) {
                                    continue;
                                    dst_y = mb_y * 16;
                                }

                                int next_start_x = (mb_x * 16) ;
                                int next_start_y = (mb_y * 16) ;
                                int next_dst_x   = (mb_x * 16) - (mx*mv_ratio)/scale;
                                int next_dst_y   = (mb_y * 16) - (my*mv_ratio)/scale;
                                if(next_start_x > (h->mb_width * 16) - 16 || next_start_x < 0) {
                                    continue;
                                    next_start_x = mb_x * 16;
                                }
                                if(next_start_y > (h->mb_height * 16) - 16 || next_start_y < 0) {
                                    continue;
                                    next_start_y = mb_y * 16;
                                }
                                if(next_dst_x > (h->mb_width * 16) - 16 || next_dst_x < 0) {
                                    continue;
                                    next_dst_x = mb_x * 16;
                                }
                                if(next_dst_y > (h->mb_height * 16) - 16 || next_dst_y < 0) {
                                    continue;
                                    next_dst_y = mb_y * 16;
                                }
                                // printf("dst_x %d, dst_y %d\n", dst_x, dst_y);
                                // printf("start_x %d, start_y %d\n", start_x, start_y);


                                vis[dst_y/8][dst_x/8] = 1;
                                vis[dst_y/8][dst_x/8 + 1] = 1;
                                vis[dst_y/8 + 1][dst_x/8] = 1;
                                vis[dst_y/8 + 1][dst_x/8 + 1] = 1;

                                mvss[dst_y/8][dst_x/8][0] = mx;
                                mvss[dst_y/8][dst_x/8][1] = my;
                                mvss[dst_y/8][dst_x/8 + 1][0] = mx;
                                mvss[dst_y/8][dst_x/8 + 1][1] = my;
                                mvss[dst_y/8 + 1][dst_x/8][0] = mx;
                                mvss[dst_y/8 + 1][dst_x/8][1] = my;
                                mvss[dst_y/8 + 1][dst_x/8 + 1][0] = mx;
                                mvss[dst_y/8 + 1][dst_x/8 + 1][1] = my;

                                h->next_vis[next_dst_y/8][next_dst_x/8] = 1;
                                h->next_vis[next_dst_y/8][next_dst_x/8 + 1] = 1;
                                h->next_vis[next_dst_y/8 + 1][next_dst_x/8] = 1;
                                h->next_vis[next_dst_y/8 + 1][next_dst_x/8 + 1] = 1;


                                int y, x, y_ref, x_ref;
                                int next_y, next_x, next_y_ref, next_x_ref;
                                for (y = start_y, y_ref = dst_y,
                                        next_y = next_start_y, next_y_ref = next_dst_y
                                        ; y < start_y + 16;
                                         y++, y_ref++,
                                         next_y++, next_y_ref++) {
                                        for (x = start_x, x_ref = dst_x,
                                        next_x = next_start_x, next_x_ref = next_dst_x;
                                         x < start_x + 16;
                                          x++, x_ref++,
                                          next_x++, next_x_ref++) {
                                            // Y
                                            interpolated_frame->data[0][y_ref * interpolated_frame->linesize[0] + x_ref]
                                            = pic->f->data[0][y * pic->f->linesize[0] + x];

                                            // next Y
                                            next_interpolated_frame->data[0][next_y_ref * next_interpolated_frame->linesize[0] + next_x_ref]
                                            = pic->f->data[0][next_y * pic->f->linesize[0] + next_x];
                                        }
                                    }
                                    for (y = start_y/2, y_ref = dst_y/2,
                                    next_y = next_start_y/2, next_y_ref = next_dst_y/2; 
                                    y < start_y/2 + (16/2);
                                     y++, y_ref++,
                                     next_y++, next_y_ref++
                                     ) {
                                        for (x = start_x/2, x_ref = dst_x/2,
                                        next_x = next_start_x/2, next_x_ref = next_dst_x/2; 
                                        x < start_x/2 + (16/2);
                                         x++, x_ref++,
                                         next_x++, next_x_ref++) {
                                            //Cb
                                            interpolated_frame->data[1][y_ref * interpolated_frame->linesize[1] + x_ref]
                                            = pic->f->data[1][y * pic->f->linesize[1] + x];
                                            
                                            // next_ Cb
                                            next_interpolated_frame->data[1][next_y_ref * next_interpolated_frame->linesize[1] + next_x_ref]
                                            = pic->f->data[1][next_y * pic->f->linesize[1] + next_x];


                                            //Cr
                                            interpolated_frame->data[2][y_ref * interpolated_frame->linesize[2] + x_ref]
                                            = pic->f->data[2][y * pic->f->linesize[2] + x];

                                            // next_ Cr
                                            next_interpolated_frame->data[2][next_y_ref * interpolated_frame->linesize[2] + next_x_ref]
                                            = pic->f->data[2][next_y * pic->f->linesize[2] + next_x];
                                        }
                                    }
                                mbcount++;
                                //mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx, my, scale, direction);
                            }
                        }
                    }
                }

/********FUTURE FRAME FILL HOLES************FUTURE FRAME FILL HOLES**********************/
/********FUTURE FRAME FILL HOLES************FUTURE FRAME FILL HOLES**********************/
                if(h->following_interpolated_frame != NULL && 0){
                    for (mb_y = 1; mb_y < h->mb_height * 2 - 1; mb_y++) {
                        for (mb_x = 1; mb_x < h->mb_width * 2 - 1; mb_x++) {
                            if(!vis[mb_y][mb_x]&&h->next_vis[mb_y][mb_x]) {
                                vis[mb_y][mb_x]=1;
                            int start_x = (mb_x * 8);
                            int start_y = (mb_y * 8);
                            int dst_x = start_x;
                            int dst_y = start_y;
                            int y_, x_, y__ref, x__ref;
                            for (y_ = start_y, y__ref = dst_y; y_ < start_y + 8; y_++, y__ref++) {
                                for (x_ = start_x, x__ref = dst_x; x_ < start_x + 8; x_++, x__ref++) {
                                    // Y
                                    interpolated_frame->data[0][y__ref * interpolated_frame->linesize[0] + x__ref]=
                                    h->following_interpolated_frame->data[0][y__ref * interpolated_frame->linesize[0] + x__ref];
                                    }
                            }
                            for (y_ = start_y/2, y__ref = dst_y/2; y_ < start_y/2 + (8/2); y_++, y__ref++) {
                                for (x_ = start_x/2, x__ref = dst_x/2; x_ < start_x/2 + (8/2); x_++, x__ref++) {
                                    //Cb
                                    interpolated_frame->data[1][y__ref * interpolated_frame->linesize[1] + x__ref]=
                                    h->following_interpolated_frame->data[1][y__ref * interpolated_frame->linesize[1] + x__ref];
                                    
                                    //Cr
                                    interpolated_frame->data[2][y__ref * interpolated_frame->linesize[2] + x__ref]=
                                    h->following_interpolated_frame->data[2][y__ref * interpolated_frame->linesize[2] + x__ref];
                                }
                            }
                            }
                        }
                    }
                }
/****END*********END********END********END***********************************************/

                int dx[8] = {-1, -1, -1, 0,  0,  1, 1, 1};
                int dy[8] = {-1,  1,  0, 1, -1, -1, 0, 1};
                for (mb_y = 1; mb_y < h->mb_height * 2 - 1; mb_y++) {
                    for (mb_x = 1; mb_x < h->mb_width * 2 - 1; mb_x++) {

                         if(!vis[mb_y][mb_x]) {
                            
                            int c = 0, sumx = 0, sumy = 0;

                             for(int k = 0; k < 8; k++){
                                int mv_x = mvss[mb_y + dy[k]][mb_x + dx[k]][0];
                                int mv_y = mvss[mb_y + dy[k]][mb_x + dx[k]][1];
                                if(mv_x != 0 || mv_y != 0){
                                    c++;
                                    sumx += mv_x;
                                    sumy += mv_y;
                                }
                            }
                            if(c != 0) {
                                sumx /= c;
                                sumy /= c;
                                vis[mb_y][mb_x] = 1;
                            }
                            int ref_frame_index = DPB_map[(pic->frame_num == 0)?max_DPB:pic->frame_num - 1];
                            // int ref_index = pic->ref_index[direction][mb_xy];
                            // int prev = (pic->ref_poc[0][direction][ref_index] - (h->slice_ctx[0].ref_list[direction][ref_index].reference & 3)) / 4;
                            // int curr = pic->frame_num;
                            // int range = (curr - prev + max_DPB + 1)%(max_DPB+1);
                            // double mv_ratio = 1-((range-0.5)/range);
                            double mv_ratio = 0.5;
                            int start_x = (mb_x * 8);
                            int start_y = (mb_y * 8);
                            int dst_x   = (mb_x * 8) + (sumx*mv_ratio)/scale;
                            int dst_y   = (mb_y * 8) + (sumy*mv_ratio)/scale;;
                            if(start_x > (h->mb_width * 16) - 8 || start_x < 0) {
                                start_x = mb_x * 8;
                            }
                            if(start_y > (h->mb_height * 16) - 8 || start_y < 0) {
                                start_y = mb_y * 8;
                            }
                            if(dst_x > (h->mb_width * 16) - 8 || dst_x < 0) {
                                dst_x = mb_x * 8;
                            }
                            if(dst_y > (h->mb_height * 16) - 8 || dst_y < 0) {
                                dst_y = mb_y * 8;
                            }
                            printf("dst_x %d, dst_y %d\n", dst_x, dst_y);
                            printf("start_x %d, start_y %d\n", start_x, start_y);

                            mvss[mb_y][mb_x][0] = sumx;
                            mvss[mb_y][mb_x][1] = sumy;

                             int y_, x_, y__ref, x__ref;
                            for (y_ = start_y, y__ref = dst_y; y_ < start_y + 8; y_++, y__ref++) {
                                for (x_ = start_x, x__ref = dst_x; x_ < start_x + 8; x_++, x__ref++) {
                                    // Y
                                    interpolated_frame->data[0][y__ref * interpolated_frame->linesize[0] + x__ref]
                                    = pic->f->data[0][y_ * pic->f->linesize[0] + x_];
                                    }
                            }
                            for (y_ = start_y/2, y__ref = dst_y/2; y_ < start_y/2 + (8/2); y_++, y__ref++) {
                                for (x_ = start_x/2, x__ref = dst_x/2; x_ < start_x/2 + (8/2); x_++, x__ref++) {
                                    //Cb
                                    interpolated_frame->data[1][y__ref * interpolated_frame->linesize[1] + x__ref]
                                    = pic->f->data[1][y_ * pic->f->linesize[1] + x_];

                                     //Cr
                                    interpolated_frame->data[2][y__ref * interpolated_frame->linesize[2] + x__ref]
                                    = pic->f->data[2][y_ * pic->f->linesize[2] + x_];
                                }
                            }
                        }
                    }
                }
                if(h->following_interpolated_frame != NULL ){
                    av_frame_free(&(h->following_interpolated_frame));
                }
                h->following_interpolated_frame = next_interpolated_frame;
                
                // printf("KKKK save frame\n");
                // printf("mbcount = %d\n", mbcount);
                save_frame_as_jpeg(avctx, interpolated_frame,  (2 * h->next_output_pic->f->coded_picture_number)-1);
            }
             // printf("KKKK finalize_frame\n");
            ret = finalize_frame(h, pict, h->next_output_pic, got_frame);
            pict->interpolated_frame = h->next_output_pic->f->interpolated_frame;
             // printf("KKKK finalize_frame finished\n");
            AVFrameSideData *sd;
            
            sd = av_frame_get_side_data(pict, AV_FRAME_DATA_MOTION_VECTORS);
            // printf("KKKK side data\n");
            // if(sd){
            //     printf("side_data\n");
            // } else {
            //     printf("no_side_data\n");
            // }
            if (ret < 0){
                // printf("ret < 0!!\n");
                return ret;
            }
            // printf("KKKK finished!!\n");
        }
    }
    // printf("KKKK before assert\n");
    av_assert0(pict->buf[0] || !*got_frame);
    // printf("KKKK after assert\n");
    ff_h264_unref_picture(h, &h->last_pic_for_ec);
    // printf("KKKK unref\n");
    return get_consumed_bytes(buf_index, buf_size);
}

#define OFFSET(x) offsetof(H264Context, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption h264_options[] = {
    { "is_avc", "is avc", OFFSET(is_avc), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, 0 },
    { "nal_length_size", "nal_length_size", OFFSET(nal_length_size), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 4, 0 },
    { "enable_er", "Enable error resilience on damaged frames (unsafe)", OFFSET(enable_er), AV_OPT_TYPE_BOOL, { .i64 = -1 }, -1, 1, VD },
    { "x264_build", "Assume this x264 version if no x264 version found in any SEI", OFFSET(x264_build), AV_OPT_TYPE_INT, {.i64 = -1}, -1, INT_MAX, VD },
    { NULL },
};

static const AVClass h264_class = {
    .class_name = "H264 Decoder",
    .item_name  = av_default_item_name,
    .option     = h264_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

AVCodec ff_h264_decoder = {
    .name                  = "h264",
    .long_name             = NULL_IF_CONFIG_SMALL("H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"),
    .type                  = AVMEDIA_TYPE_VIDEO,
    .id                    = AV_CODEC_ID_H264,
    .priv_data_size        = sizeof(H264Context),
    .init                  = h264_decode_init,
    .close                 = h264_decode_end,
    .decode                = h264_decode_frame,
    .capabilities          = /*AV_CODEC_CAP_DRAW_HORIZ_BAND |*/ AV_CODEC_CAP_DR1 |
                             AV_CODEC_CAP_DELAY | AV_CODEC_CAP_SLICE_THREADS |
                             AV_CODEC_CAP_FRAME_THREADS,
    .hw_configs            = (const AVCodecHWConfigInternal*[]) {
#if CONFIG_H264_DXVA2_HWACCEL
                               HWACCEL_DXVA2(h264),
#endif
#if CONFIG_H264_D3D11VA_HWACCEL
                               HWACCEL_D3D11VA(h264),
#endif
#if CONFIG_H264_D3D11VA2_HWACCEL
                               HWACCEL_D3D11VA2(h264),
#endif
#if CONFIG_H264_NVDEC_HWACCEL
                               HWACCEL_NVDEC(h264),
#endif
#if CONFIG_H264_VAAPI_HWACCEL
                               HWACCEL_VAAPI(h264),
#endif
#if CONFIG_H264_VDPAU_HWACCEL
                               HWACCEL_VDPAU(h264),
#endif
#if CONFIG_H264_VIDEOTOOLBOX_HWACCEL
                               HWACCEL_VIDEOTOOLBOX(h264),
#endif
                               NULL
                           },
    .caps_internal         = FF_CODEC_CAP_INIT_THREADSAFE | FF_CODEC_CAP_EXPORTS_CROPPING,
    .flush                 = flush_dpb,
    .init_thread_copy      = ONLY_IF_THREADS_ENABLED(decode_init_thread_copy),
    .update_thread_context = ONLY_IF_THREADS_ENABLED(ff_h264_update_thread_context),
    .profiles              = NULL_IF_CONFIG_SMALL(ff_h264_profiles),
    .priv_class            = &h264_class,
};
