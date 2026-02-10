/*
 * neuron-shim: ONNX Runtime backend
 *
 * This is the preferred backend for GPU-accelerated inference.
 * ONNX Runtime provides a single C API that automatically dispatches
 * to the best available hardware:
 *
 *   NVIDIA GPU  → CUDAExecutionProvider (or TensorRT EP)
 *   AMD GPU     → MIGraphXExecutionProvider (replaces ROCm EP)
 *   No GPU      → CPUExecutionProvider (fallback)
 *
 * The application doesn't need to know which GPU vendor is present —
 * ORT handles it via execution provider priority.
 *
 * Model format: .onnx
 *   Conversion chain: .dla → .tflite → .onnx (via tf2onnx)
 *
 * Dependencies:
 *   - libonnxruntime.so (core, always needed)
 *   - NVIDIA: CUDA toolkit + cuDNN (auto-detected by ORT)
 *   - AMD: ROCm + MIGraphX (auto-detected by ORT)
 *
 * Docker images:
 *   NVIDIA: nvcr.io/nvidia/tritonserver or onnxruntime-gpu pip package
 *   AMD:    rocm/onnxruntime or build from source with --use_rocm
 *
 * Compile with: -lonnxruntime
 */

#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* ONNX Runtime C API                                                  */
/* We use the C API (not C++) for maximum ABI stability and to avoid   */
/* linking issues across different ORT builds.                         */
/* ------------------------------------------------------------------ */
#include <onnxruntime_c_api.h>

#define MAX_TENSORS 32

/* Helper macro for ORT error checking */
#define ORT_CHECK(api, expr) \
    do { \
        OrtStatus* _s = (expr); \
        if (_s) { \
            fprintf(stderr, "[neuron-shim][onnx] ERROR: %s\n", \
                    (api)->GetErrorMessage(_s)); \
            (api)->ReleaseStatus(_s); \
            return -1; \
        } \
    } while(0)

typedef struct {
    const OrtApi*       api;
    OrtEnv*             env;
    OrtSessionOptions*  session_opts;
    OrtSession*         session;
    OrtMemoryInfo*      memory_info;

    /* Input tensor metadata (populated after model load) */
    struct {
        char     name[256];
        size_t   size;        /* total byte size */
        int64_t  shape[8];
        size_t   num_dims;
        ONNXTensorElementDataType type;
    } inputs[MAX_TENSORS];
    size_t input_count;

    /* Output tensor metadata */
    struct {
        char     name[256];
        size_t   size;
        int64_t  shape[8];
        size_t   num_dims;
        ONNXTensorElementDataType type;
    } outputs[MAX_TENSORS];
    size_t output_count;

    /* User-bound I/O buffers */
    struct {
        const void* buf;
        size_t      size;
    } input_bindings[MAX_TENSORS];

    struct {
        void*  buf;
        size_t size;
    } output_bindings[MAX_TENSORS];

} OnnxContext;

/* ------------------------------------------------------------------ */
/* Helper: get byte size of an ORT tensor element type                 */
/* ------------------------------------------------------------------ */
static size_t ort_element_size(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:   return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:     return 1;
        default: return 4;
    }
}

/* ------------------------------------------------------------------ */
/* Helper: compute total byte size from shape + element type           */
/* ------------------------------------------------------------------ */
static size_t compute_tensor_size(const int64_t* shape, size_t num_dims,
                                   ONNXTensorElementDataType type) {
    size_t total = ort_element_size(type);
    for (size_t i = 0; i < num_dims; i++) {
        int64_t d = shape[i];
        if (d <= 0) d = 1; /* dynamic dims treated as 1 for sizing */
        total *= (size_t)d;
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
static int onnx_create(void** ctx) {
    OnnxContext* c = (OnnxContext*)calloc(1, sizeof(OnnxContext));
    if (!c) return -1;

    c->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!c->api) {
        fprintf(stderr, "[neuron-shim][onnx] failed to get ORT API v%d\n",
                ORT_API_VERSION);
        free(c);
        return -1;
    }

    /* Create environment */
    ORT_CHECK(c->api,
        c->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "neuron-shim", &c->env));

    /* Session options — add execution providers in priority order */
    ORT_CHECK(c->api,
        c->api->CreateSessionOptions(&c->session_opts));

    /* Set thread count */
    int num_threads = 4;
    const char* env_threads = getenv("NEURON_SHIM_NUM_THREADS");
    if (env_threads) num_threads = atoi(env_threads);
    c->api->SetIntraOpNumThreads(c->session_opts, num_threads);

    /* Enable graph optimizations */
    c->api->SetSessionGraphOptimizationLevel(c->session_opts,
                                              ORT_ENABLE_ALL);

    /*
     * Execution provider registration.
     *
     * We try to register GPU providers. If they're not available
     * (library not compiled with that EP), the call fails silently
     * and ORT falls back to CPU.
     *
     * Priority: TensorRT > CUDA > MIGraphX > CPU
     */
    const char* force_cpu = getenv("NEURON_SHIM_FORCE_CPU");
    if (!force_cpu || strcmp(force_cpu, "1") != 0) {

        /* Try NVIDIA TensorRT (best perf on NVIDIA) */
        {
            OrtTensorRTProviderOptionsV2* trt_opts = NULL;
            OrtStatus* s = c->api->CreateTensorRTProviderOptions(&trt_opts);
            if (!s && trt_opts) {
                s = c->api->SessionOptionsAppendExecutionProvider_TensorRT_V2(
                        c->session_opts, trt_opts);
                if (!s) {
                    fprintf(stderr, "[neuron-shim][onnx] TensorRT EP: registered\n");
                } else {
                    c->api->ReleaseStatus(s);
                }
                c->api->ReleaseTensorRTProviderOptions(trt_opts);
            } else if (s) {
                c->api->ReleaseStatus(s);
            }
        }

        /* Try NVIDIA CUDA */
        {
            OrtCUDAProviderOptionsV2* cuda_opts = NULL;
            OrtStatus* s = c->api->CreateCUDAProviderOptions(&cuda_opts);
            if (!s && cuda_opts) {
                s = c->api->SessionOptionsAppendExecutionProvider_CUDA_V2(
                        c->session_opts, cuda_opts);
                if (!s) {
                    fprintf(stderr, "[neuron-shim][onnx] CUDA EP: registered\n");
                } else {
                    c->api->ReleaseStatus(s);
                }
                c->api->ReleaseCUDAProviderOptions(cuda_opts);
            } else if (s) {
                c->api->ReleaseStatus(s);
            }
        }

        /* Try AMD MIGraphX (replaces ROCm EP from ORT 1.23+) */
        {
            OrtStatus* s = OrtSessionOptionsAppendExecutionProvider_MIGraphX(
                    c->session_opts, 0 /* device_id */);
            if (!s) {
                fprintf(stderr, "[neuron-shim][onnx] MIGraphX EP: registered\n");
            } else {
                c->api->ReleaseStatus(s);
                /* Fallback: try legacy ROCm EP for older ORT versions */
                s = OrtSessionOptionsAppendExecutionProvider_ROCM(
                        c->session_opts, 0);
                if (!s) {
                    fprintf(stderr, "[neuron-shim][onnx] ROCm EP: registered\n");
                } else {
                    c->api->ReleaseStatus(s);
                }
            }
        }
    }

    /* CPU is always available as final fallback */
    fprintf(stderr, "[neuron-shim][onnx] CPU EP: always available\n");

    /* Memory info for tensor creation */
    ORT_CHECK(c->api,
        c->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                     &c->memory_info));

    *ctx = c;
    return 0;
}

static void onnx_destroy(void* ctx) {
    OnnxContext* c = (OnnxContext*)ctx;
    if (!c) return;

    if (c->session)      c->api->ReleaseSession(c->session);
    if (c->session_opts) c->api->ReleaseSessionOptions(c->session_opts);
    if (c->memory_info)  c->api->ReleaseMemoryInfo(c->memory_info);
    if (c->env)          c->api->ReleaseEnv(c->env);
    free(c);
}

/* ------------------------------------------------------------------ */
/* Helper: populate tensor metadata from a loaded session              */
/* ------------------------------------------------------------------ */
static int populate_tensor_info(OnnxContext* c) {
    OrtAllocator* allocator;
    ORT_CHECK(c->api, c->api->GetAllocatorWithDefaultOptions(&allocator));

    /* Inputs */
    ORT_CHECK(c->api, c->api->SessionGetInputCount(c->session, &c->input_count));
    if (c->input_count > MAX_TENSORS) c->input_count = MAX_TENSORS;

    for (size_t i = 0; i < c->input_count; i++) {
        char* name;
        ORT_CHECK(c->api, c->api->SessionGetInputName(c->session, i,
                                                        allocator, &name));
        snprintf(c->inputs[i].name, sizeof(c->inputs[i].name), "%s", name);
        allocator->Free(allocator, name);

        OrtTypeInfo* type_info;
        ORT_CHECK(c->api, c->api->SessionGetInputTypeInfo(c->session, i,
                                                            &type_info));

        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_CHECK(c->api, c->api->CastTypeInfoToTensorInfo(type_info,
                                                             &tensor_info));

        ORT_CHECK(c->api, c->api->GetTensorElementType(tensor_info,
                                                         &c->inputs[i].type));
        ORT_CHECK(c->api, c->api->GetDimensionsCount(tensor_info,
                                                       &c->inputs[i].num_dims));
        ORT_CHECK(c->api, c->api->GetDimensions(tensor_info,
                                                  c->inputs[i].shape,
                                                  c->inputs[i].num_dims));

        c->inputs[i].size = compute_tensor_size(c->inputs[i].shape,
                                                 c->inputs[i].num_dims,
                                                 c->inputs[i].type);

        c->api->ReleaseTypeInfo(type_info);

        fprintf(stderr, "[neuron-shim][onnx] input[%zu]: '%s' %zu bytes\n",
                i, c->inputs[i].name, c->inputs[i].size);
    }

    /* Outputs */
    ORT_CHECK(c->api, c->api->SessionGetOutputCount(c->session, &c->output_count));
    if (c->output_count > MAX_TENSORS) c->output_count = MAX_TENSORS;

    for (size_t i = 0; i < c->output_count; i++) {
        char* name;
        ORT_CHECK(c->api, c->api->SessionGetOutputName(c->session, i,
                                                         allocator, &name));
        snprintf(c->outputs[i].name, sizeof(c->outputs[i].name), "%s", name);
        allocator->Free(allocator, name);

        OrtTypeInfo* type_info;
        ORT_CHECK(c->api, c->api->SessionGetOutputTypeInfo(c->session, i,
                                                             &type_info));

        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_CHECK(c->api, c->api->CastTypeInfoToTensorInfo(type_info,
                                                             &tensor_info));

        ORT_CHECK(c->api, c->api->GetTensorElementType(tensor_info,
                                                         &c->outputs[i].type));
        ORT_CHECK(c->api, c->api->GetDimensionsCount(tensor_info,
                                                       &c->outputs[i].num_dims));
        ORT_CHECK(c->api, c->api->GetDimensions(tensor_info,
                                                  c->outputs[i].shape,
                                                  c->outputs[i].num_dims));

        c->outputs[i].size = compute_tensor_size(c->outputs[i].shape,
                                                  c->outputs[i].num_dims,
                                                  c->outputs[i].type);

        c->api->ReleaseTypeInfo(type_info);

        fprintf(stderr, "[neuron-shim][onnx] output[%zu]: '%s' %zu bytes\n",
                i, c->outputs[i].name, c->outputs[i].size);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Model loading                                                       */
/* ------------------------------------------------------------------ */
static int onnx_load_from_file(void* ctx, const char* path) {
    OnnxContext* c = (OnnxContext*)ctx;

    fprintf(stderr, "[neuron-shim][onnx] loading: %s\n", path);

    ORT_CHECK(c->api,
        c->api->CreateSession(c->env, path, c->session_opts, &c->session));

    return populate_tensor_info(c);
}

static int onnx_load_from_buffer(void* ctx, const void* buf, size_t size) {
    OnnxContext* c = (OnnxContext*)ctx;

    fprintf(stderr, "[neuron-shim][onnx] loading from buffer: %zu bytes\n", size);

    ORT_CHECK(c->api,
        c->api->CreateSessionFromArray(c->env, buf, size,
                                        c->session_opts, &c->session));

    return populate_tensor_info(c);
}

/* ------------------------------------------------------------------ */
/* Tensor metadata queries                                             */
/* ------------------------------------------------------------------ */
static int onnx_get_input_count(void* ctx, uint32_t* count) {
    OnnxContext* c = (OnnxContext*)ctx;
    *count = (uint32_t)c->input_count;
    return 0;
}

static int onnx_get_output_count(void* ctx, uint32_t* count) {
    OnnxContext* c = (OnnxContext*)ctx;
    *count = (uint32_t)c->output_count;
    return 0;
}

static int onnx_get_input_size(void* ctx, int index, size_t* size) {
    OnnxContext* c = (OnnxContext*)ctx;
    if ((size_t)index >= c->input_count) return -1;
    *size = c->inputs[index].size;
    return 0;
}

static int onnx_get_output_size(void* ctx, int index, size_t* size) {
    OnnxContext* c = (OnnxContext*)ctx;
    if ((size_t)index >= c->output_count) return -1;
    *size = c->outputs[index].size;
    return 0;
}

/* ------------------------------------------------------------------ */
/* I/O binding                                                         */
/* ------------------------------------------------------------------ */
static int onnx_set_input(void* ctx, int index, const void* buf, size_t size) {
    OnnxContext* c = (OnnxContext*)ctx;
    if ((size_t)index >= MAX_TENSORS) return -1;
    c->input_bindings[index].buf  = buf;
    c->input_bindings[index].size = size;
    return 0;
}

static int onnx_set_output(void* ctx, int index, void* buf, size_t size) {
    OnnxContext* c = (OnnxContext*)ctx;
    if ((size_t)index >= MAX_TENSORS) return -1;
    c->output_bindings[index].buf  = buf;
    c->output_bindings[index].size = size;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */
static int onnx_invoke(void* ctx) {
    OnnxContext* c = (OnnxContext*)ctx;
    if (!c->session) return -1;

    /*
     * Build input tensors from bound buffers.
     * ORT wants arrays of names and OrtValue pointers.
     */
    const char*  input_names[MAX_TENSORS];
    OrtValue*    input_tensors[MAX_TENSORS];

    for (size_t i = 0; i < c->input_count; i++) {
        input_names[i] = c->inputs[i].name;

        ORT_CHECK(c->api,
            c->api->CreateTensorWithDataAsOrtValue(
                c->memory_info,
                (void*)c->input_bindings[i].buf,
                c->input_bindings[i].size,
                c->inputs[i].shape,
                c->inputs[i].num_dims,
                c->inputs[i].type,
                &input_tensors[i]));
    }

    /* Prepare output names */
    const char*  output_names[MAX_TENSORS];
    OrtValue*    output_tensors[MAX_TENSORS];

    for (size_t i = 0; i < c->output_count; i++) {
        output_names[i] = c->outputs[i].name;
        output_tensors[i] = NULL; /* ORT allocates these */
    }

    /* Run inference */
    ORT_CHECK(c->api,
        c->api->Run(c->session, NULL,
                     input_names, (const OrtValue* const*)input_tensors,
                     c->input_count,
                     output_names, c->output_count,
                     output_tensors));

    /* Copy outputs to user buffers */
    for (size_t i = 0; i < c->output_count; i++) {
        if (output_tensors[i] && c->output_bindings[i].buf) {
            void* tensor_data;
            ORT_CHECK(c->api,
                c->api->GetTensorMutableData(output_tensors[i], &tensor_data));

            size_t copy_size = c->output_bindings[i].size;
            if (copy_size > c->outputs[i].size)
                copy_size = c->outputs[i].size;

            memcpy(c->output_bindings[i].buf, tensor_data, copy_size);
        }
    }

    /* Cleanup */
    for (size_t i = 0; i < c->input_count; i++)
        c->api->ReleaseValue(input_tensors[i]);
    for (size_t i = 0; i < c->output_count; i++)
        if (output_tensors[i]) c->api->ReleaseValue(output_tensors[i]);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Backend vtable                                                      */
/* ------------------------------------------------------------------ */
static const NeuronShimBackend onnx_backend = {
    .name             = "onnx",
    .create           = onnx_create,
    .destroy          = onnx_destroy,
    .load_from_file   = onnx_load_from_file,
    .load_from_buffer = onnx_load_from_buffer,
    .get_input_count  = onnx_get_input_count,
    .get_output_count = onnx_get_output_count,
    .get_input_size   = onnx_get_input_size,
    .get_output_size  = onnx_get_output_size,
    .set_input        = onnx_set_input,
    .set_output       = onnx_set_output,
    .invoke           = onnx_invoke,
};

const NeuronShimBackend* neuron_shim_backend_onnx(void) {
    return &onnx_backend;
}
