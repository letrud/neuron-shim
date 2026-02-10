/*
 * neuron-shim: Model path resolver
 *
 *   model_dir empty:  /path/to/model.dla → /path/to/model.dla.onnx
 *   model_dir set:    /path/to/model.dla → <model_dir>/model.dla.onnx
 */

#ifndef NEURON_SHIM_MODEL_RESOLVER_H
#define NEURON_SHIM_MODEL_RESOLVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Resolve a model path by appending suffix (and optionally redirecting).
 *
 * @param original_path  Path the application requested
 * @param suffix         Suffix to append (e.g. ".onnx")
 * @param model_dir      Override directory (NULL or "" = use original dir)
 * @param resolved       Output buffer
 * @param resolved_len   Size of output buffer
 *
 * @return 0 if resolved file exists, -1 with error printed if not
 */
int neuron_shim_resolve_model(const char* original_path,
                               const char* suffix,
                               const char* model_dir,
                               char* resolved,
                               size_t resolved_len);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_SHIM_MODEL_RESOLVER_H */
