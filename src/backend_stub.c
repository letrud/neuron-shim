/*
 * neuron-shim: Stub backend
 *
 * Returns success for every call, outputs zeroed buffers.
 * Useful for:
 *   - Testing if the application boots without real inference
 *   - Tracing which API calls the application makes
 *   - Identifying the model files and I/O shapes needed
 */

#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 32

typedef struct {
    /* Track what the app loaded so we can log it */
    char model_path[1024];

    /* Fake tensor sizes — set from setInput/setOutput calls */
    struct { size_t size; } inputs[MAX_TENSORS];
    struct { size_t size; void* buf; } outputs[MAX_TENSORS];
    int input_count;
    int output_count;
    int inference_count;
} StubContext;

static int stub_create(void** ctx) {
    StubContext* c = (StubContext*)calloc(1, sizeof(StubContext));
    *ctx = c;
    fprintf(stderr, "[neuron-shim][stub] backend created — all calls are no-ops\n");
    return 0;
}

static void stub_destroy(void* ctx) {
    StubContext* c = (StubContext*)ctx;
    if (c) {
        fprintf(stderr, "[neuron-shim][stub] stats: %d inferences on '%s'\n",
                c->inference_count, c->model_path);
    }
    free(c);
}

static int stub_load_from_file(void* ctx, const char* path) {
    StubContext* c = (StubContext*)ctx;
    snprintf(c->model_path, sizeof(c->model_path), "%s", path);
    fprintf(stderr, "[neuron-shim][stub] LOAD: %s\n", path);
    /* Default to 1 input, 1 output until we see actual calls */
    c->input_count = 1;
    c->output_count = 1;
    return 0;
}

static int stub_load_from_buffer(void* ctx, const void* buf, size_t size) {
    StubContext* c = (StubContext*)ctx;
    snprintf(c->model_path, sizeof(c->model_path), "<buffer:%zu bytes>", size);
    fprintf(stderr, "[neuron-shim][stub] LOAD from buffer: %zu bytes\n", size);
    c->input_count = 1;
    c->output_count = 1;
    return 0;
}

static int stub_get_input_count(void* ctx, uint32_t* count) {
    StubContext* c = (StubContext*)ctx;
    *count = c->input_count > 0 ? c->input_count : 1;
    return 0;
}

static int stub_get_output_count(void* ctx, uint32_t* count) {
    StubContext* c = (StubContext*)ctx;
    *count = c->output_count > 0 ? c->output_count : 1;
    return 0;
}

static int stub_get_input_size(void* ctx, int index, size_t* size) {
    StubContext* c = (StubContext*)ctx;
    *size = (index < MAX_TENSORS && c->inputs[index].size > 0)
            ? c->inputs[index].size : 1024;
    return 0;
}

static int stub_get_output_size(void* ctx, int index, size_t* size) {
    StubContext* c = (StubContext*)ctx;
    *size = (index < MAX_TENSORS && c->outputs[index].size > 0)
            ? c->outputs[index].size : 1024;
    return 0;
}

static int stub_set_input(void* ctx, int index, const void* buf, size_t size) {
    StubContext* c = (StubContext*)ctx;
    if (index < MAX_TENSORS) {
        c->inputs[index].size = size;
        if (index >= c->input_count) c->input_count = index + 1;
    }
    fprintf(stderr, "[neuron-shim][stub] SET_INPUT[%d]: %zu bytes\n", index, size);
    return 0;
}

static int stub_set_output(void* ctx, int index, void* buf, size_t size) {
    StubContext* c = (StubContext*)ctx;
    if (index < MAX_TENSORS) {
        c->outputs[index].size = size;
        c->outputs[index].buf  = buf;
        if (index >= c->output_count) c->output_count = index + 1;
    }
    fprintf(stderr, "[neuron-shim][stub] SET_OUTPUT[%d]: %zu bytes\n", index, size);
    return 0;
}

static int stub_invoke(void* ctx) {
    StubContext* c = (StubContext*)ctx;
    c->inference_count++;

    /* Zero all output buffers — this gives "no detections" for most
     * object detection models, which is the safest default */
    for (int i = 0; i < c->output_count && i < MAX_TENSORS; i++) {
        if (c->outputs[i].buf && c->outputs[i].size > 0) {
            memset(c->outputs[i].buf, 0, c->outputs[i].size);
        }
    }

    if (c->inference_count <= 5 || (c->inference_count % 100) == 0) {
        fprintf(stderr, "[neuron-shim][stub] INFERENCE #%d (outputs zeroed)\n",
                c->inference_count);
    }

    return 0;
}

static const NeuronShimBackend stub_backend = {
    .name             = "stub",
    .create           = stub_create,
    .destroy          = stub_destroy,
    .load_from_file   = stub_load_from_file,
    .load_from_buffer = stub_load_from_buffer,
    .get_input_count  = stub_get_input_count,
    .get_output_count = stub_get_output_count,
    .get_input_size   = stub_get_input_size,
    .get_output_size  = stub_get_output_size,
    .set_input        = stub_set_input,
    .set_output       = stub_set_output,
    .invoke           = stub_invoke,
};

const NeuronShimBackend* neuron_shim_backend_stub(void) {
    return &stub_backend;
}
