/*
 * neuron-shim: Backend abstraction
 *
 * Each backend implements this interface. The shim selects which
 * backend to use at runtime based on the NEURON_SHIM_BACKEND env var
 * or auto-detects from available libraries.
 */

#ifndef NEURON_SHIM_BACKEND_H
#define NEURON_SHIM_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Backend interface — every backend implements these                   */
/* ------------------------------------------------------------------ */
typedef struct NeuronShimBackend {
    const char* name;

    /* Lifecycle */
    int  (*create)(void** ctx);
    void (*destroy)(void* ctx);

    /* Model loading.
     * 'path' may point to a .dla — the backend is responsible for
     * resolving this to its native format (e.g. .tflite).
     * See model_resolver.h for the lookup logic. */
    int  (*load_from_file)(void* ctx, const char* path);
    int  (*load_from_buffer)(void* ctx, const void* buf, size_t size);

    /* Tensor metadata */
    int  (*get_input_count)(void* ctx, uint32_t* count);
    int  (*get_output_count)(void* ctx, uint32_t* count);
    int  (*get_input_size)(void* ctx, int index, size_t* size);
    int  (*get_output_size)(void* ctx, int index, size_t* size);

    /* I/O binding */
    int  (*set_input)(void* ctx, int index, const void* buf, size_t size);
    int  (*set_output)(void* ctx, int index, void* buf, size_t size);

    /* Inference */
    int  (*invoke)(void* ctx);

} NeuronShimBackend;

/* ------------------------------------------------------------------ */
/* Built-in backends                                                   */
/* ------------------------------------------------------------------ */
extern const NeuronShimBackend* neuron_shim_backend_tflite(void);
extern const NeuronShimBackend* neuron_shim_backend_onnx(void);
extern const NeuronShimBackend* neuron_shim_backend_stub(void);

/* Select backend by name or auto-detect */
const NeuronShimBackend* neuron_shim_select_backend(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_SHIM_BACKEND_H */
