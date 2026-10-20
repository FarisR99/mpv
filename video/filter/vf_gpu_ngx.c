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

#include <windows.h>

#include <libplacebo/vulkan.h>

#include "common/common.h"
#include "common/msg.h"
#include "video/filter/vf_gpu_ngx.h"

#define FFNV_LOG_FUNC(logctx, ...) mp_msg((logctx), MSGL_ERR, __VA_ARGS__)
#define FFNV_DEBUG_LOG_FUNC(logctx, ...) mp_msg((logctx), MSGL_DEBUG, __VA_ARGS__)
#include <ffnvcodec/dynlink_loader.h>

// The C API of NVEncNVSDKNGX.dll. The functions return 0 on success.
#define NGX_DLL_NAME L"NVEncNVSDKNGX.dll"
#define NGX_FEATURE_TRUEHDR 2

struct ngx_rect {
    int left, top, right, bottom;
};

typedef int (__stdcall *ngx_create_fn)(void **handle, int feature);
typedef int (__stdcall *ngx_init_fn)(void *handle, int cuda_device_ordinal,
                                     void *cu_context, void *cu_stream);
typedef void (__stdcall *ngx_delete_fn)(void *handle);
typedef int (__stdcall *ngx_proc_frame_fn)(void *handle,
                                           const struct ngx_rect *rect_dst,
                                           const struct ngx_rect *rect_src,
                                           const void *params,
                                           const void *src, int src_pitch,
                                           void *dst, int dst_pitch,
                                           int src_bytes_per_pixel,
                                           int dst_bytes_per_pixel);

// A libplacebo texture shared with CUDA, plus a linear copy of it: the DLL
// takes device pointers rather than arrays.
struct shared_tex {
    pl_tex tex;
    CUexternalMemory mem;
    CUmipmappedArray mma;
    CUarray array;
    pl_vulkan_sem sem;
    HANDLE sem_handle;
    CUexternalSemaphore cu_sem;
    CUdeviceptr buf;
    size_t pitch;
    int bytes_per_pixel;
};

struct vf_gpu_ngx {
    struct mp_log *log;
    pl_gpu gpu;
    struct vf_gpu_ngx_params params;

    CudaFunctions *cu;
    CUdevice dev;
    CUcontext ctx; // the device's primary context, as the DLL's cudart uses

    HMODULE dll;
    ngx_create_fn create;
    ngx_init_fn init;
    ngx_delete_fn delete;
    ngx_proc_frame_fn proc_frame;
    void *handle;

    int w, h;
    struct shared_tex in;  // RGBA8 SDR
    struct shared_tex out; // RGBA16F linear light
};

static bool check_cu(struct vf_gpu_ngx *ngx, CUresult err, const char *func)
{
    if (err == CUDA_SUCCESS)
        return true;
    const char *name = NULL, *desc = NULL;
    ngx->cu->cuGetErrorName(err, &name);
    ngx->cu->cuGetErrorString(err, &desc);
    MP_ERR(ngx, "%s failed: %s: %s\n", func, name ? name : "?", desc ? desc : "?");
    return false;
}

#define CHECK_CU(x) check_cu(ngx, (x), #x)

static bool push_ctx(struct vf_gpu_ngx *ngx)
{
    return CHECK_CU(ngx->cu->cuCtxPushCurrent(ngx->ctx));
}

static void pop_ctx(struct vf_gpu_ngx *ngx)
{
    CUcontext dummy;
    CHECK_CU(ngx->cu->cuCtxPopCurrent(&dummy));
}

static void shared_tex_uninit(struct vf_gpu_ngx *ngx, struct shared_tex *st)
{
    CudaFunctions *cu = ngx->cu;
    if (st->buf)
        CHECK_CU(cu->cuMemFree(st->buf));
    if (st->mma)
        CHECK_CU(cu->cuMipmappedArrayDestroy(st->mma));
    if (st->mem)
        CHECK_CU(cu->cuDestroyExternalMemory(st->mem));
    if (st->cu_sem)
        CHECK_CU(cu->cuDestroyExternalSemaphore(st->cu_sem));
    pl_vulkan_sem_destroy(ngx->gpu, &st->sem.sem);
    if (st->sem_handle)
        CloseHandle(st->sem_handle);
    pl_tex_destroy(ngx->gpu, &st->tex);
    *st = (struct shared_tex){0};
}

static bool shared_tex_init(struct vf_gpu_ngx *ngx, struct shared_tex *st,
                            enum pl_fmt_type type, int depth,
                            CUarray_format cu_format, bool renderable)
{
    CudaFunctions *cu = ngx->cu;
    int w = ngx->w, h = ngx->h;

    pl_fmt fmt = pl_find_fmt(ngx->gpu, type, 4, depth, depth,
                             renderable ? PL_FMT_CAP_RENDERABLE
                                        : PL_FMT_CAP_SAMPLEABLE);
    if (!fmt) {
        MP_ERR(ngx, "No suitable %d-bit RGBA texture format.\n", depth);
        return false;
    }
    st->bytes_per_pixel = fmt->texel_size;
    st->tex = pl_tex_create(ngx->gpu, pl_tex_params(
        .w = w,
        .h = h,
        .format = fmt,
        .sampleable = !renderable,
        .renderable = renderable,
        .export_handle = PL_HANDLE_WIN32,
    ));
    if (!st->tex)
        return false;

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC mem_desc = {
        .type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
        .handle.win32.handle = st->tex->shared_mem.handle.handle,
        .size = st->tex->shared_mem.size,
    };
    if (!CHECK_CU(cu->cuImportExternalMemory(&st->mem, &mem_desc)))
        return false;

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC array_desc = {
        .offset = st->tex->shared_mem.offset,
        .arrayDesc = {
            .Width = w,
            .Height = h,
            .Format = cu_format,
            .NumChannels = 4,
        },
        .numLevels = 1,
    };
    if (!CHECK_CU(cu->cuExternalMemoryGetMappedMipmappedArray(&st->mma, st->mem,
                                                              &array_desc)) ||
        !CHECK_CU(cu->cuMipmappedArrayGetLevel(&st->array, st->mma, 0)))
        return false;

    union pl_handle sem_handle = {0};
    st->sem.sem = pl_vulkan_sem_create(ngx->gpu, pl_vulkan_sem_params(
        .type = VK_SEMAPHORE_TYPE_TIMELINE,
        .export_handle = PL_HANDLE_WIN32,
        .out_handle = &sem_handle,
    ));
    if (!st->sem.sem)
        return false;
    // Unlike memory, CUDA doesn't take ownership of the semaphore handle.
    st->sem_handle = sem_handle.handle;

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sem_desc = {
        .type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
        .handle.win32.handle = st->sem_handle,
    };
    if (!CHECK_CU(cu->cuImportExternalSemaphore(&st->cu_sem, &sem_desc)))
        return false;

    return CHECK_CU(cu->cuMemAllocPitch(&st->buf, &st->pitch,
                                        w * st->bytes_per_pixel, h, 16));
}

static void free_textures(struct vf_gpu_ngx *ngx)
{
    if (!ngx->in.tex && !ngx->out.tex)
        return;
    // Neither API may still be using the textures.
    pl_gpu_finish(ngx->gpu);
    if (push_ctx(ngx)) {
        CHECK_CU(ngx->cu->cuCtxSynchronize());
        shared_tex_uninit(ngx, &ngx->in);
        shared_tex_uninit(ngx, &ngx->out);
        pop_ctx(ngx);
    }
    ngx->w = ngx->h = 0;
}

void vf_gpu_ngx_destroy(struct vf_gpu_ngx **ngx_ptr)
{
    struct vf_gpu_ngx *ngx = *ngx_ptr;
    if (!ngx)
        return;

    if (ngx->cu) {
        free_textures(ngx);
        if (ngx->handle && push_ctx(ngx)) {
            ngx->delete(ngx->handle);
            pop_ctx(ngx);
        }
        if (ngx->ctx)
            CHECK_CU(ngx->cu->cuDevicePrimaryCtxRelease(ngx->dev));
    }
    // Unloading also resets the DLL's once-per-process NGX initialization, so
    // a later instance can initialize it again.
    if (ngx->dll)
        FreeLibrary(ngx->dll);
    cuda_free_functions(&ngx->cu);
    TA_FREEP(ngx_ptr);
}

static bool init_cuda(struct vf_gpu_ngx *ngx)
{
    if (cuda_load_functions(&ngx->cu, ngx->log) < 0)
        return false;
    CudaFunctions *cu = ngx->cu;
    if (!cu->cuImportExternalMemory) {
        MP_ERR(ngx, "TrueHDR requires a newer NVIDIA driver.\n");
        return false;
    }

    int count = 0;
    if (!CHECK_CU(cu->cuInit(0)) || !CHECK_CU(cu->cuDeviceGetCount(&count)))
        return false;
    for (int i = 0; i < count; i++) {
        CUdevice dev;
        CUuuid uuid;
        if (cu->cuDeviceGet(&dev, i) != CUDA_SUCCESS ||
            cu->cuDeviceGetUuid(&uuid, dev) != CUDA_SUCCESS)
            continue;
        if (memcmp(ngx->gpu->uuid, uuid.bytes, sizeof(ngx->gpu->uuid)) == 0) {
            ngx->dev = dev;
            return CHECK_CU(cu->cuDevicePrimaryCtxRetain(&ngx->ctx, dev));
        }
    }
    MP_ERR(ngx, "The Vulkan device is not a CUDA device; TrueHDR requires an "
           "NVIDIA RTX GPU.\n");
    return false;
}

static bool init_ngx(struct vf_gpu_ngx *ngx)
{
    ngx->dll = LoadLibraryW(NGX_DLL_NAME);
    if (!ngx->dll) {
        MP_ERR(ngx, "Could not load NVEncNVSDKNGX.dll. Place it, "
               "nvngx_truehdr.dll and cudart64_12.dll next to mpv.\n");
        return false;
    }

#define LOAD(field, type, name) \
    ngx->field = (type)(void *)GetProcAddress(ngx->dll, name); \
    if (!ngx->field) { \
        MP_ERR(ngx, "NVEncNVSDKNGX.dll has no %s.\n", name); \
        return false; \
    }
    LOAD(create, ngx_create_fn, "NVEncNVSDKNGXCreate");
    LOAD(init, ngx_init_fn, "NVEncNVSDKNGXInit");
    LOAD(delete, ngx_delete_fn, "NVEncNVSDKNGXDelete");
    LOAD(proc_frame, ngx_proc_frame_fn, "NVEncNVSDKNGXProcFrame");
#undef LOAD

    int err = ngx->create(&ngx->handle, NGX_FEATURE_TRUEHDR);
    if (err || !ngx->handle) {
        MP_ERR(ngx, "Could not create the TrueHDR feature (error %d).\n", err);
        return false;
    }

    if (!push_ctx(ngx))
        return false;
    // Everything runs on the context's legacy default stream, which the
    // DLL's own copies use too, so all of it is ordered.
    err = ngx->init(ngx->handle, -1, ngx->ctx, NULL);
    pop_ctx(ngx);
    if (err) {
        MP_ERR(ngx, "Could not initialize TrueHDR (error %d). It requires "
               "nvngx_truehdr.dll and an RTX GPU.\n", err);
        return false;
    }
    return true;
}

struct vf_gpu_ngx *vf_gpu_ngx_create(struct mp_log *log, pl_gpu gpu,
                                     const struct vf_gpu_ngx_params *params)
{
    struct vf_gpu_ngx *ngx = talloc_zero(NULL, struct vf_gpu_ngx);
    ngx->log = log;
    ngx->gpu = gpu;
    ngx->params = *params;

    if (!pl_vulkan_get(gpu)) {
        MP_ERR(ngx, "TrueHDR requires api=vulkan.\n");
        goto error;
    }
    if (!(gpu->export_caps.tex & PL_HANDLE_WIN32) ||
        !(gpu->export_caps.sync & PL_HANDLE_WIN32))
    {
        MP_ERR(ngx, "TrueHDR requires exportable Vulkan textures and "
               "semaphores.\n");
        goto error;
    }

    if (!init_cuda(ngx) || !init_ngx(ngx))
        goto error;
    return ngx;

error:
    vf_gpu_ngx_destroy(&ngx);
    return NULL;
}

pl_tex vf_gpu_ngx_get_input(struct vf_gpu_ngx *ngx, int w, int h)
{
    if (ngx->in.tex && ngx->w == w && ngx->h == h)
        return ngx->in.tex;

    free_textures(ngx);
    ngx->w = w;
    ngx->h = h;
    if (!push_ctx(ngx))
        return NULL;
    bool ok = shared_tex_init(ngx, &ngx->in, PL_FMT_UNORM, 8,
                              CU_AD_FORMAT_UNSIGNED_INT8, true) &&
              shared_tex_init(ngx, &ngx->out, PL_FMT_FLOAT, 16,
                              CU_AD_FORMAT_HALF, false);
    pop_ctx(ngx);
    if (!ok) {
        MP_ERR(ngx, "Could not share %dx%d textures with CUDA.\n", w, h);
        free_textures(ngx);
        return NULL;
    }
    return ngx->in.tex;
}

static bool copy_array(struct vf_gpu_ngx *ngx, struct shared_tex *st,
                       bool to_buf)
{
    CUDA_MEMCPY2D copy = {
        .WidthInBytes = ngx->w * st->bytes_per_pixel,
        .Height = ngx->h,
    };
    if (to_buf) {
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.srcArray = st->array;
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.dstDevice = st->buf;
        copy.dstPitch = st->pitch;
    } else {
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = st->buf;
        copy.srcPitch = st->pitch;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = st->array;
    }
    return CHECK_CU(ngx->cu->cuMemcpy2DAsync(&copy, 0));
}

pl_tex vf_gpu_ngx_process(struct vf_gpu_ngx *ngx)
{
    CudaFunctions *cu = ngx->cu;
    struct shared_tex *st[2] = {&ngx->in, &ngx->out};
    if (!ngx->in.tex)
        return NULL;

    // Hand both textures to CUDA: the input once it is rendered, and the
    // output once the previous frame's YUV conversion has read it.
    for (int n = 0; n < 2; n++) {
        st[n]->sem.value++;
        if (!pl_vulkan_hold_ex(ngx->gpu, pl_vulkan_hold_params(
                .tex = st[n]->tex,
                .layout = VK_IMAGE_LAYOUT_GENERAL,
                .qf = VK_QUEUE_FAMILY_EXTERNAL,
                .semaphore = st[n]->sem,
            )))
            return NULL;
    }

    bool ok = push_ctx(ngx);
    if (ok) {
        CUexternalSemaphore sems[2] = {ngx->in.cu_sem, ngx->out.cu_sem};
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait[2] = {
            {.params.fence.value = ngx->in.sem.value},
            {.params.fence.value = ngx->out.sem.value},
        };
        ok = CHECK_CU(cu->cuWaitExternalSemaphoresAsync(sems, wait, 2, 0)) &&
             copy_array(ngx, &ngx->in, true);

        if (ok) {
            const struct ngx_rect rect = {0, 0, ngx->w, ngx->h};
            int err = ngx->proc_frame(ngx->handle, &rect, &rect, &ngx->params,
                                      (void *)(uintptr_t)ngx->in.buf,
                                      ngx->in.pitch,
                                      (void *)(uintptr_t)ngx->out.buf,
                                      ngx->out.pitch,
                                      ngx->in.bytes_per_pixel,
                                      ngx->out.bytes_per_pixel);
            if (err)
                MP_ERR(ngx, "TrueHDR failed (error %d).\n", err);
            ok = !err && copy_array(ngx, &ngx->out, false);
        }

        // Signal even after errors, so Vulkan isn't left waiting forever.
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal[2] = {
            {.params.fence.value = ++ngx->in.sem.value},
            {.params.fence.value = ++ngx->out.sem.value},
        };
        ok = CHECK_CU(cu->cuSignalExternalSemaphoresAsync(sems, signal, 2, 0)) && ok;
        pop_ctx(ngx);
    }

    for (int n = 0; n < 2; n++) {
        pl_vulkan_release_ex(ngx->gpu, pl_vulkan_release_params(
            .tex = st[n]->tex,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .qf = VK_QUEUE_FAMILY_EXTERNAL,
            .semaphore = st[n]->sem,
        ));
    }
    return ok ? ngx->out.tex : NULL;
}
