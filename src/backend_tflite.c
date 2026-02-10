/*
 * neuron-shim: TensorFlow Lite backend
 *
 * This is the main backend for actual inference. It loads .tflite
 * models and runs them on CPU (or GPU delegate if available).
 *
 * Compile with: -ltensorflowlite_c
 * (uses the TFLite C API for maximum portability)
 */

#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* TFLite C API — we use the stable C interface, not C++               */
/* This avoids ABI issues and works with any TFLite build.             */
/* ------------------------------------------------------------------ */
#include <tensorflow/lite/c/c_api.h>

/* Optional: GPU delegate for acceleration */
#ifdef NEURON_SHIM_ENABLE_GPU
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

#define MAX_TENSORS 32

typedef struct {
    TfLiteModel*       model;
    TfLiteInterpreter* interpreter;
    TfLiteInterpreterOptions* options;

    /* User-provided output buffers to copy results into */
    struct {
        void*  buf;
        size_t size;
    } output_bindings[MAX_TENSORS];
    int output_binding_count;
} TFLiteContext;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
static int tflite_create(void** ctx) {
    TFLiteContext* c = (TFLiteContext*)calloc(1, sizeof(TFLiteContext));
    if (!c) return -1;

    c->options = TfLiteInterpreterOptionsCreate();

    /* Use all available cores */
    int num_threads = 4;
    const char* env = getenv("NEURON_SHIM_NUM_THREADS");
    if (env) num_threads = atoi(env);
    TfLiteInterpreterOptionsSetNumThreads(c->options, num_threads);

#ifdef NEURON_SHIM_ENABLE_GPU
    {
        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        if (gpu_delegate) {
            TfLiteInterpreterOptionsAddDelegate(c->options, gpu_delegate);
            fprintf(stderr, "[neuron-shim][tflite] GPU delegate enabled\n");
        }
    }
#endif

    *ctx = c;
    return 0;
}

static void tflite_destroy(void* ctx) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c) return;

    if (c->interpreter)
        TfLiteInterpreterDelete(c->interpreter);
    if (c->model)
        TfLiteModelDelete(c->model);
    if (c->options)
        TfLiteInterpreterOptionsDelete(c->options);

    free(c);
}

/* ------------------------------------------------------------------ */
/* Model loading                                                       */
/* ------------------------------------------------------------------ */
static int tflite_build_interpreter(TFLiteContext* c) {
    if (!c->model) return -1;

    c->interpreter = TfLiteInterpreterCreate(c->model, c->options);
    if (!c->interpreter) {
        fprintf(stderr, "[neuron-shim][tflite] failed to create interpreter\n");
        return -1;
    }

    if (TfLiteInterpreterAllocateTensors(c->interpreter) != kTfLiteOk) {
        fprintf(stderr, "[neuron-shim][tflite] AllocateTensors failed\n");
        return -1;
    }

    fprintf(stderr, "[neuron-shim][tflite] model loaded: %d inputs, %d outputs\n",
            TfLiteInterpreterGetInputTensorCount(c->interpreter),
            TfLiteInterpreterGetOutputTensorCount(c->interpreter));

    return 0;
}

static int tflite_load_from_file(void* ctx, const char* path) {
    TFLiteContext* c = (TFLiteContext*)ctx;

    c->model = TfLiteModelCreateFromFile(path);
    if (!c->model) {
        fprintf(stderr, "[neuron-shim][tflite] failed to load: %s\n", path);
        return -1;
    }

    return tflite_build_interpreter(c);
}

static int tflite_load_from_buffer(void* ctx, const void* buf, size_t size) {
    TFLiteContext* c = (TFLiteContext*)ctx;

    c->model = TfLiteModelCreate(buf, size);
    if (!c->model) {
        fprintf(stderr, "[neuron-shim][tflite] failed to load from buffer\n");
        return -1;
    }

    return tflite_build_interpreter(c);
}

/* ------------------------------------------------------------------ */
/* Tensor metadata                                                     */
/* ------------------------------------------------------------------ */
static int tflite_get_input_count(void* ctx, uint32_t* count) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c->interpreter) return -1;
    *count = (uint32_t)TfLiteInterpreterGetInputTensorCount(c->interpreter);
    return 0;
}

static int tflite_get_output_count(void* ctx, uint32_t* count) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c->interpreter) return -1;
    *count = (uint32_t)TfLiteInterpreterGetOutputTensorCount(c->interpreter);
    return 0;
}

static int tflite_get_input_size(void* ctx, int index, size_t* size) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c->interpreter) return -1;

    const TfLiteTensor* tensor =
        TfLiteInterpreterGetInputTensor(c->interpreter, index);
    if (!tensor) return -1;

    *size = TfLiteTensorByteSize(tensor);
    return 0;
}

static int tflite_get_output_size(void* ctx, int index, size_t* size) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c->interpreter) return -1;

    const TfLiteTensor* tensor =
        TfLiteInterpreterGetOutputTensor(c->interpreter, index);
    if (!tensor) return -1;

    *size = TfLiteTensorByteSize(tensor);
    return 0;
}

/* ------------------------------------------------------------------ */
/* I/O binding                                                         */
/* ------------------------------------------------------------------ */
static int tflite_set_input(void* ctx, int index, const void* buf, size_t size) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c->interpreter) return -1;

    TfLiteTensor* tensor =
        TfLiteInterpreterGetInputTensor(c->interpreter, index);
    if (!tensor) return -1;

    if (TfLiteTensorCopyFromBuffer(tensor, buf, size) != kTfLiteOk)
        return -1;

    return 0;
}

static int tflite_set_output(void* ctx, int index, void* buf, size_t size) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (index >= MAX_TENSORS) return -1;

    /* Store binding — we'll copy data here after inference */
    c->output_bindings[index].buf  = buf;
    c->output_bindings[index].size = size;
    if (index >= c->output_binding_count)
        c->output_binding_count = index + 1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */
static int tflite_invoke(void* ctx) {
    TFLiteContext* c = (TFLiteContext*)ctx;
    if (!c->interpreter) return -1;

    if (TfLiteInterpreterInvoke(c->interpreter) != kTfLiteOk) {
        fprintf(stderr, "[neuron-shim][tflite] inference failed\n");
        return -1;
    }

    /* Copy output data to user-provided buffers */
    for (int i = 0; i < c->output_binding_count; i++) {
        if (!c->output_bindings[i].buf) continue;

        const TfLiteTensor* tensor =
            TfLiteInterpreterGetOutputTensor(c->interpreter, i);
        if (!tensor) continue;

        size_t copy_size = c->output_bindings[i].size;
        size_t tensor_size = TfLiteTensorByteSize(tensor);
        if (copy_size > tensor_size) copy_size = tensor_size;

        TfLiteTensorCopyToBuffer(tensor,
                                 c->output_bindings[i].buf,
                                 copy_size);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Backend vtable                                                      */
/* ------------------------------------------------------------------ */
static const NeuronShimBackend tflite_backend = {
    .name             = "tflite",
    .create           = tflite_create,
    .destroy          = tflite_destroy,
    .load_from_file   = tflite_load_from_file,
    .load_from_buffer = tflite_load_from_buffer,
    .get_input_count  = tflite_get_input_count,
    .get_output_count = tflite_get_output_count,
    .get_input_size   = tflite_get_input_size,
    .get_output_size  = tflite_get_output_size,
    .set_input        = tflite_set_input,
    .set_output       = tflite_set_output,
    .invoke           = tflite_invoke,
};

const NeuronShimBackend* neuron_shim_backend_tflite(void) {
    return &tflite_backend;
}
