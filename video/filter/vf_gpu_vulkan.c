/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/version.h>

#include "options/m_config.h"
#include "video/filter/vf_gpu.h"
#include "video/mp_image.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/placebo/utils.h"
#include "video/out/vulkan/context.h"
#include "video/out/vulkan/utils.h"

// Hardware output needs AVVulkanDeviceContext.qf.
#define VF_GPU_VK_HW_OUTPUT (LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(59, 34, 100))

struct vk_offscreen_ctx {
    struct ra_ctx *ractx;
    // Not a talloc child of the offscreen_ctx: once av_device exists, the
    // device owns it, so that encoders holding our frames can outlive us.
    struct mpvk_ctx *vk;
#if VF_GPU_VK_HW_OUTPUT
    AVBufferRef *av_device;
    AVBufferRef *hw_pool[2]; // per get_hw_frame layout
    // The frame from get_hw_frame that libplacebo is rendering into.
    AVHWFramesContext *cur_hwfc;
    AVVkFrame *cur_vkf;
    pl_tex cur_tex[MP_MAX_PLANES];
    int cur_num_planes;
    bool cur_multiplane;
#endif
};

extern const struct m_sub_options vulkan_conf;

static void free_vk(struct mpvk_ctx *vk)
{
    pl_vulkan_destroy(&vk->vulkan);
    mpvk_uninit(vk);
    talloc_free(vk);
}

#if VF_GPU_VK_HW_OUTPUT
static void lock_queue(struct AVHWDeviceContext *ctx,
                       uint32_t queue_family, uint32_t index)
{
    struct mpvk_ctx *vk = ctx->user_opaque;
    vk->vulkan->lock_queue(vk->vulkan, queue_family, index);
}

static void unlock_queue(struct AVHWDeviceContext *ctx,
                         uint32_t queue_family, uint32_t index)
{
    struct mpvk_ctx *vk = ctx->user_opaque;
    vk->vulkan->unlock_queue(vk->vulkan, queue_family, index);
}

static void free_av_device(struct AVHWDeviceContext *ctx)
{
    free_vk(ctx->user_opaque);
}

// Wrap libplacebo's device for FFmpeg, as hwdec_vulkan does, but with the
// video encode queues for encoders such as hevc_vulkan.
static bool init_av_device(struct offscreen_ctx *ctx)
{
    struct vk_offscreen_ctx *p = ctx->priv;
    struct mpvk_ctx *vk = p->vk;
    if (p->av_device)
        return true;

    uint32_t num_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(vk->vulkan->phys_device, &num_qf, NULL);
    VkQueueFamilyProperties2 *qf = talloc_array(NULL, VkQueueFamilyProperties2, num_qf);
    VkQueueFamilyVideoPropertiesKHR *qf_vid =
        talloc_array(qf, VkQueueFamilyVideoPropertiesKHR, num_qf);
    for (int i = 0; i < num_qf; i++) {
        qf_vid[i] = (VkQueueFamilyVideoPropertiesKHR) {
            .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR,
        };
        qf[i] = (VkQueueFamilyProperties2) {
            .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
            .pNext = &qf_vid[i],
        };
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(vk->vulkan->phys_device, &num_qf, qf);

    AVBufferRef *ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!ref)
        goto error;
    AVHWDeviceContext *device_ctx = (void *)ref->data;
    AVVulkanDeviceContext *hwctx = device_ctx->hwctx;

    device_ctx->user_opaque = vk;
#if LIBAVUTIL_VERSION_MAJOR < 62
    AV_NOWARN_DEPRECATED(hwctx->lock_queue = lock_queue;)
    AV_NOWARN_DEPRECATED(hwctx->unlock_queue = unlock_queue;)
#endif
    hwctx->get_proc_addr = vk->vkinst->get_proc_addr;
    hwctx->inst = vk->vkinst->instance;
    hwctx->phys_dev = vk->vulkan->phys_device;
    hwctx->act_dev = vk->vulkan->device;
    hwctx->device_features = *vk->vulkan->features;
    hwctx->enabled_inst_extensions = vk->vkinst->extensions;
    hwctx->nb_enabled_inst_extensions = vk->vkinst->num_extensions;
    hwctx->enabled_dev_extensions = vk->vulkan->extensions;
    hwctx->nb_enabled_dev_extensions = vk->vulkan->num_extensions;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(60, 32, 100) && PL_API_VER >= 365
    hwctx->queue_flags = vk->vulkan->queue_graphics.flags;
#endif

    hwctx->nb_qf = 0;
    hwctx->qf[hwctx->nb_qf++] = (AVVulkanDeviceQueueFamily) {
        .idx = vk->vulkan->queue_graphics.index,
        .num = vk->vulkan->queue_graphics.count,
        .flags = VK_QUEUE_GRAPHICS_BIT,
    };
    hwctx->qf[hwctx->nb_qf++] = (AVVulkanDeviceQueueFamily) {
        .idx = vk->vulkan->queue_transfer.index,
        .num = vk->vulkan->queue_transfer.count,
        .flags = VK_QUEUE_TRANSFER_BIT,
    };
    hwctx->qf[hwctx->nb_qf++] = (AVVulkanDeviceQueueFamily) {
        .idx = vk->vulkan->queue_compute.index,
        .num = vk->vulkan->queue_compute.count,
        .flags = VK_QUEUE_COMPUTE_BIT,
    };
    // mppl_create_vulkan requests queues in every encode family.
    for (int i = 0; i < num_qf && hwctx->nb_qf < MP_ARRAY_SIZE(hwctx->qf); i++) {
        VkQueueFlags flags = qf[i].queueFamilyProperties.queueFlags;
        if (flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
            hwctx->qf[hwctx->nb_qf++] = (AVVulkanDeviceQueueFamily) {
                .idx = i,
                .num = qf[i].queueFamilyProperties.queueCount,
                .flags = VK_QUEUE_VIDEO_ENCODE_BIT_KHR,
                .video_caps = qf_vid[i].videoCodecOperations,
            };
        }
    }

    if (av_hwdevice_ctx_init(ref) < 0) {
        mp_verbose(ctx->log, "Could not create an FFmpeg Vulkan device.\n");
        av_buffer_unref(&ref);
        goto error;
    }
    // Set only now: the free callback runs on failed init too.
    device_ctx->free = free_av_device;
    p->av_device = ref;
    talloc_free(qf);
    return true;

error:
    talloc_free(qf);
    return false;
}

static bool init_hw_pool(struct offscreen_ctx *ctx, enum AVPixelFormat sw_format,
                         int layout, int w, int h)
{
    struct vk_offscreen_ctx *p = ctx->priv;
    AVBufferRef **pool = &p->hw_pool[layout];
    if (*pool) {
        AVHWFramesContext *frames = (void *)(*pool)->data;
        if (frames->sw_format == sw_format && frames->width == w &&
            frames->height == h)
            return true;
        av_buffer_unref(pool);
    }

    if (!init_av_device(ctx))
        return false;
    *pool = av_hwframe_ctx_alloc(p->av_device);
    if (!*pool)
        return false;

    AVHWFramesContext *frames = (void *)(*pool)->data;
    frames->format = AV_PIX_FMT_VULKAN;
    frames->sw_format = sw_format;
    frames->width = w;
    frames->height = h;
    // A dynamic pool gives each in-flight encoder frame a separate image.
    frames->initial_pool_size = 0;
    AVVulkanFramesContext *vk_frames = frames->hwctx;
    // FFmpeg adds the usage it needs (sampling, storage, encoding).
    vk_frames->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (layout == 1)
        vk_frames->flags |= AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE;
    if (av_hwframe_ctx_init(*pool) < 0) {
        av_buffer_unref(pool);
        return false;
    }
    return true;
}

// Take the frame back from libplacebo and record its state for the consumer.
static void end_hw_frame(struct offscreen_ctx *ctx)
{
    struct vk_offscreen_ctx *p = ctx->priv;
    AVVkFrame *vkf = p->cur_vkf;
    if (!vkf)
        return;

    for (int i = 0; i < p->cur_num_planes; i++) {
        int index = p->cur_multiplane ? 0 : i;
        // All planes end up in the same layout, as AVVkFrame records one per
        // image.
        bool ok = pl_vulkan_hold_ex(p->vk->gpu, pl_vulkan_hold_params(
            .tex = p->cur_tex[i],
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .qf = VK_QUEUE_FAMILY_IGNORED,
            .semaphore = (pl_vulkan_sem) {
                .sem = vkf->sem[index],
                .value = vkf->sem_value[index] + 1,
            },
        ));
        if (ok) {
            vkf->layout[index] = VK_IMAGE_LAYOUT_GENERAL;
            vkf->access[index] = 0;
            vkf->sem_value[index]++;
        }
        p->cur_tex[i] = NULL;
    }

    const AVVulkanFramesContext *vk_frames = p->cur_hwfc->hwctx;
    vk_frames->unlock_frame(p->cur_hwfc, vkf);
    p->cur_vkf = NULL;
    p->cur_hwfc = NULL;
}

static struct mp_image *get_hw_frame(struct offscreen_ctx *ctx, int imgfmt,
                                    int layout, int w, int h,
                                    pl_tex planes[MP_MAX_PLANES])
{
    struct vk_offscreen_ctx *p = ctx->priv;
    pl_gpu gpu = p->vk->gpu;
    struct mp_image *img = NULL;
    AVFrame *frame = NULL;
    int num_planes = 0;
    for (int n = 0; n < MP_MAX_PLANES; n++)
        planes[n] = NULL;
    mp_assert(!p->cur_vkf);

    enum AVPixelFormat sw_format;
    switch (imgfmt) {
    case IMGFMT_BGRA:
        sw_format = AV_PIX_FMT_BGRA;
        num_planes = 1;
        break;
    case IMGFMT_NV12:
        if ((w | h) & 1)
            return NULL;
        sw_format = AV_PIX_FMT_NV12;
        num_planes = 2;
        break;
    default:
        return NULL;
    }

    if (!init_hw_pool(ctx, sw_format, layout, w, h))
        goto error;
    frame = av_frame_alloc();
    if (!frame || av_hwframe_get_buffer(p->hw_pool[layout], frame, 0) < 0)
        goto error;
    frame->sample_aspect_ratio = (AVRational){1, 1};
    AVHWFramesContext *hwfc = (void *)frame->hw_frames_ctx->data;
    const AVVulkanFramesContext *vk_frames = hwfc->hwctx;
    AVVkFrame *vkf = (AVVkFrame *)frame->data[0];
    img = mp_image_from_av_frame(frame);
    av_frame_free(&frame);
    if (!img)
        goto error;

    int num_images = 0;
    while (num_images < AV_NUM_DATA_POINTERS && vkf->img[num_images])
        num_images++;
    bool multiplane = num_planes > 1 && num_images == 1;
    const VkFormat *vk_fmt = av_vkfmt_from_pixfmt(sw_format);
    // FFmpeg stores BGRA in R8G8B8A8 images and handles the byte order
    // itself. Render through a B8G8R8A8 view (the images have a mutable
    // format), so the bytes end up in BGRA order.
    static const VkFormat bgra_fmt[] = {VK_FORMAT_B8G8R8A8_UNORM};
    if (sw_format == AV_PIX_FMT_BGRA)
        vk_fmt = bgra_fmt;
    static const VkImageAspectFlags plane_aspect[] = {
        VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT,
    };

    for (int n = 0; n < num_planes; n++) {
        // The wrapped image starts out held by us.
        planes[n] = pl_vulkan_wrap(gpu, pl_vulkan_wrap_params(
            .image = vkf->img[multiplane ? 0 : n],
            .width = n ? w / 2 : w,
            .height = n ? h / 2 : h,
            .format = vk_fmt[n],
            .usage = vk_frames->usage,
            .aspect = multiplane ? plane_aspect[n] : VK_IMAGE_ASPECT_COLOR_BIT,
        ));
        if (!planes[n] || !planes[n]->params.renderable)
            goto error;
    }

    // Hand the planes to libplacebo until end_hw_frame. FFmpeg's frame lock
    // stays held meanwhile, as required around command submission.
    vk_frames->lock_frame(hwfc, vkf);
    for (int n = 0; n < num_planes; n++) {
        int index = multiplane ? 0 : n;
        pl_vulkan_release_ex(gpu, pl_vulkan_release_params(
            .tex = planes[n],
            .layout = vkf->layout[index],
            .qf = VK_QUEUE_FAMILY_IGNORED,
            .semaphore = (pl_vulkan_sem) {
                .sem = vkf->sem[index],
                .value = vkf->sem_value[index],
            },
        ));
        p->cur_tex[n] = planes[n];
    }
    p->cur_hwfc = hwfc;
    p->cur_vkf = vkf;
    p->cur_num_planes = num_planes;
    p->cur_multiplane = multiplane;
    return img;

error:
    mp_verbose(ctx->log, "Could not allocate a Vulkan %s hardware render target.\n",
               mp_imgfmt_to_name(imgfmt));
    for (int n = 0; n < num_planes; n++)
        pl_tex_destroy(gpu, &planes[n]);
    av_frame_free(&frame);
    talloc_free(img);
    return NULL;
}
#endif

static void vk_ctx_destroy(void *p)
{
    struct offscreen_ctx *ctx = p;
    struct vk_offscreen_ctx *vkctx = ctx->priv;
    // Creation failed before priv was set; the error path cleaned up already.
    if (!vkctx)
        return;
    struct ra_ctx *ractx = vkctx->ractx;
    struct mpvk_ctx *vk = vkctx->vk;

#if VF_GPU_VK_HW_OUTPUT
    end_hw_frame(ctx);
    for (int n = 0; n < MP_ARRAY_SIZE(vkctx->hw_pool); n++)
        av_buffer_unref(&vkctx->hw_pool[n]);
#endif

    if (ractx->ra) {
        pl_gpu_finish(vk->gpu);
        ractx->ra->fns->destroy(ctx->ra);
        ractx->ra = NULL;
        ctx->ra = NULL;
    }

    vk->gpu = NULL;
#if VF_GPU_VK_HW_OUTPUT
    // Frames still queued in an encoder keep the device, and with it vk, alive.
    if (vkctx->av_device) {
        av_buffer_unref(&vkctx->av_device);
        return;
    }
#endif
    free_vk(vk);
}

static struct offscreen_ctx *vk_offscreen_ctx_create(struct mpv_global *global,
                                                     struct mp_log *log)
{
    struct offscreen_ctx *ctx = talloc(NULL, struct offscreen_ctx);
    talloc_set_destructor(ctx, vk_ctx_destroy);
    *ctx = (struct offscreen_ctx){
        .log = log,
    };

    struct ra_ctx *ractx = talloc_zero(ctx, struct ra_ctx);
    struct mpvk_ctx *vk = talloc_zero(NULL, struct mpvk_ctx);
    ractx->log = ctx->log;
    ractx->global = global;

    vk->pllog = mppl_log_create(vk, log);
    if (!vk->pllog)
        goto error;

    struct pl_vk_inst_params pl_vk_params = {0};
    struct ra_ctx_opts *ctx_opts = mp_get_config_group(ctx, global, &ra_ctx_conf);
    pl_vk_params.debug = ctx_opts->debug;
    mppl_log_set_probing(vk->pllog, true);
    vk->vkinst = pl_vk_inst_create(vk->pllog, &pl_vk_params);
    mppl_log_set_probing(vk->pllog, false);
    if (!vk->vkinst)
        goto error;

    struct vulkan_opts *vk_opts = mp_get_config_group(ctx, global, &vulkan_conf);
    vk->vulkan = mppl_create_vulkan(vk_opts, vk->vkinst, vk->pllog, VK_NULL_HANDLE,
                                    ctx_opts->allow_sw);
    if (!vk->vulkan)
        goto error;

    vk->gpu = vk->vulkan->gpu;
    ractx->ra = ra_create_pl(vk->gpu, ractx->log);
    if (!ractx->ra)
        goto error;

    struct vk_offscreen_ctx *vkctx = talloc_zero(ctx, struct vk_offscreen_ctx);
    *vkctx = (struct vk_offscreen_ctx){
        .ractx = ractx,
        .vk = vk
    };

    ctx->ra = ractx->ra;
    ctx->priv = vkctx;
#if VF_GPU_VK_HW_OUTPUT
    ctx->get_hw_frame = get_hw_frame;
    ctx->num_hw_layouts = 2;
    ctx->end_hw_frame = end_hw_frame;
#endif

    talloc_free(ctx_opts);
    talloc_free(vk_opts);
    return ctx;

error:
    free_vk(vk);
    talloc_free(ctx);
    return NULL;
}

const struct offscreen_context offscreen_vk = {
    .api = "vulkan",
    .offscreen_ctx_create = vk_offscreen_ctx_create
};
