/*
 * neuron-shim: Core runtime shim
 *
 * Drop-in replacement for MediaTek libneuronrt.so.
 * Redirects NeuronRuntime_* calls through configurable backends.
 *
 * Configuration (in priority order):
 *   1. Environment variables (NEURON_SHIM_*)
 *   2. ./neuron-shim.conf
 *   3. /etc/neuron-shim.conf
 *   4. Defaults
 *
 * Model resolution:
 *   app loads "model.dla" → shim loads "model.dla.onnx"
 *   (suffix is configurable)
 */

#include "RuntimeAPI.h"
#include "backend.h"
#include "config.h"
#include "model_resolver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* ------------------------------------------------------------------ */
/* Logging                                                             */
/* ------------------------------------------------------------------ */
static int g_log_level = 3;

#define SHIM_LOG(level, tag, fmt, ...) \
    do { \
        if (g_log_level >= (level)) \
            fprintf(stderr, "[neuron-shim][%s] " fmt "\n", \
                    (tag), ##__VA_ARGS__); \
    } while(0)

#define LOG_ERR(fmt, ...)   SHIM_LOG(1, "ERROR", fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  SHIM_LOG(2, "WARN",  fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  SHIM_LOG(3, "INFO",  fmt, ##__VA_ARGS__)
#define LOG_DBG(fmt, ...)   SHIM_LOG(4, "DEBUG", fmt, ##__VA_ARGS__)

/* ------------------------------------------------------------------ */
/* Internal runtime context                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    const NeuronShimBackend* backend;
    void*                    backend_ctx;
} ShimRuntime;

/* ------------------------------------------------------------------ */
/* Global state                                                        */
/* ------------------------------------------------------------------ */
static const NeuronShimConfig*  g_config  = NULL;
static const NeuronShimBackend* g_backend = NULL;
static const char*              g_suffix  = NULL;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

static void shim_global_init(void) {
    /* Load config from files + env */
    g_config = neuron_shim_config_load();
    g_log_level = g_config->log_level;

    LOG_INFO("=== neuron-shim initializing ===");
    LOG_INFO("config: backend=%s suffix=%s threads=%d force_cpu=%d",
             g_config->backend, g_config->suffix,
             g_config->threads, g_config->force_cpu);
    if (g_config->model_dir[0] != '\0')
        LOG_INFO("config: model_dir=%s", g_config->model_dir);

    /* Select backend */
    g_backend = neuron_shim_select_backend(
        strcmp(g_config->backend, "auto") == 0 ? NULL : g_config->backend);
    LOG_INFO("active backend: %s", g_backend->name);

    /* Resolve suffix */
    g_suffix = neuron_shim_config_get_suffix(g_config);
    LOG_INFO("model suffix: %s", g_suffix);
    if (g_config->model_dir[0] != '\0')
        LOG_INFO("model resolution: <path>.dla → %s/<basename>.dla%s",
                 g_config->model_dir, g_suffix);
    else
        LOG_INFO("model resolution: <path>.dla → <path>.dla%s", g_suffix);
}

static inline void ensure_init(void) {
    pthread_once(&g_init_once, shim_global_init);
}

/* ------------------------------------------------------------------ */
/* NeuronRuntime_create                                                */
/* ------------------------------------------------------------------ */
int NeuronRuntime_create(const RuntimeConfig* config, NeuronRuntime* runtime) {
    ensure_init();
    (void)config;

    ShimRuntime* rt = (ShimRuntime*)calloc(1, sizeof(ShimRuntime));
    if (!rt) {
        LOG_ERR("alloc failed");
        return NEURONRUNTIME_OP_FAILED;
    }

    rt->backend = g_backend;

    int err = rt->backend->create(&rt->backend_ctx);
    if (err != 0) {
        LOG_ERR("backend create failed: %d", err);
        free(rt);
        return NEURONRUNTIME_OP_FAILED;
    }

    *(ShimRuntime**)runtime = rt;
    LOG_DBG("runtime created: %p", (void*)rt);
    return NEURONRUNTIME_NO_ERROR;
}

/* ------------------------------------------------------------------ */
/* NeuronRuntime_release                                               */
/* ------------------------------------------------------------------ */
int NeuronRuntime_release(NeuronRuntime runtime) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt) return NEURONRUNTIME_UNEXPECTED_NULL;

    LOG_DBG("runtime release: %p", (void*)rt);
    rt->backend->destroy(rt->backend_ctx);
    free(rt);
    return NEURONRUNTIME_NO_ERROR;
}

/* ------------------------------------------------------------------ */
/* Model loading                                                       */
/* ------------------------------------------------------------------ */
int NeuronRuntime_loadNetworkFromFile(NeuronRuntime runtime, const char* path) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !path) return NEURONRUNTIME_UNEXPECTED_NULL;

    LOG_INFO("loadNetwork: %s", path);

    /* Resolve: model.dla → model.dla.onnx (or redirect via model_dir) */
    char resolved[1024];
    int rc = neuron_shim_resolve_model(path, g_suffix, g_config->model_dir,
                                        resolved, sizeof(resolved));
    if (rc != 0) {
        LOG_ERR("model not found: %s%s", path, g_suffix);
        return NEURONRUNTIME_BAD_DATA;
    }

    LOG_INFO("loading: %s", resolved);
    int ret = rt->backend->load_from_file(rt->backend_ctx, resolved);
    if (ret != 0) {
        LOG_ERR("backend failed to load: %s", resolved);
        return NEURONRUNTIME_OP_FAILED;
    }

    return NEURONRUNTIME_NO_ERROR;
}

int NeuronRuntime_loadNetworkFromBuffer(NeuronRuntime runtime,
                                        const void* buffer, size_t size) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !buffer) return NEURONRUNTIME_UNEXPECTED_NULL;

    LOG_INFO("loadNetworkFromBuffer: %zu bytes", size);
    int ret = rt->backend->load_from_buffer(rt->backend_ctx, buffer, size);
    return ret == 0 ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

/* ------------------------------------------------------------------ */
/* Input / Output                                                      */
/* ------------------------------------------------------------------ */
int NeuronRuntime_setInput(NeuronRuntime runtime,
                           int index, const void* buffer,
                           size_t size, int padding) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt) return NEURONRUNTIME_UNEXPECTED_NULL;
    (void)padding;
    LOG_DBG("setInput[%d] %zu bytes", index, size);
    return rt->backend->set_input(rt->backend_ctx, index, buffer, size) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_setOutput(NeuronRuntime runtime,
                            int index, void* buffer,
                            size_t size, int padding) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt) return NEURONRUNTIME_UNEXPECTED_NULL;
    (void)padding;
    LOG_DBG("setOutput[%d] %zu bytes", index, size);
    return rt->backend->set_output(rt->backend_ctx, index, buffer, size) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_getInputCount(NeuronRuntime runtime, uint32_t* count) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !count) return NEURONRUNTIME_UNEXPECTED_NULL;
    return rt->backend->get_input_count(rt->backend_ctx, count) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_getOutputCount(NeuronRuntime runtime, uint32_t* count) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !count) return NEURONRUNTIME_UNEXPECTED_NULL;
    return rt->backend->get_output_count(rt->backend_ctx, count) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_getInputSize(NeuronRuntime runtime, int index, size_t* size) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !size) return NEURONRUNTIME_UNEXPECTED_NULL;
    return rt->backend->get_input_size(rt->backend_ctx, index, size) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_getOutputSize(NeuronRuntime runtime, int index, size_t* size) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !size) return NEURONRUNTIME_UNEXPECTED_NULL;
    return rt->backend->get_output_size(rt->backend_ctx, index, size) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_getInputInfo(NeuronRuntime runtime,
                               int index, NeuronTensorInfo* info) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !info) return NEURONRUNTIME_UNEXPECTED_NULL;
    memset(info, 0, sizeof(*info));
    return rt->backend->get_input_size(rt->backend_ctx, index, &info->sizeBytes) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

int NeuronRuntime_getOutputInfo(NeuronRuntime runtime,
                                int index, NeuronTensorInfo* info) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt || !info) return NEURONRUNTIME_UNEXPECTED_NULL;
    memset(info, 0, sizeof(*info));
    return rt->backend->get_output_size(rt->backend_ctx, index, &info->sizeBytes) == 0
        ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */
int NeuronRuntime_inference(NeuronRuntime runtime) {
    ShimRuntime* rt = (ShimRuntime*)runtime;
    if (!rt) return NEURONRUNTIME_UNEXPECTED_NULL;

    LOG_DBG("inference begin");
    int ret = rt->backend->invoke(rt->backend_ctx);
    LOG_DBG("inference done: %d", ret);
    return ret == 0 ? NEURONRUNTIME_NO_ERROR : NEURONRUNTIME_OP_FAILED;
}

/* ------------------------------------------------------------------ */
/* QoS — all no-ops                                                    */
/* ------------------------------------------------------------------ */
int NeuronRuntime_setQoSOption(NeuronRuntime runtime, const QoSOptions* qos) {
    (void)runtime; (void)qos;
    return NEURONRUNTIME_NO_ERROR;
}

int NeuronRuntime_getProfiledQoSData(NeuronRuntime runtime, QoSOptions* qos) {
    (void)runtime;
    if (qos) {
        qos->profiledQoSData = NULL;
        qos->profiledQoSDataSize = 0;
    }
    return NEURONRUNTIME_NO_ERROR;
}
