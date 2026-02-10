/*
 * neuron-shim: Model path resolver
 *
 * Resolution strategy:
 *
 *   If model_dir is set:
 *     /usr/share/models/person_detect.dla
 *     → <model_dir>/person_detect.dla.onnx
 *
 *   If model_dir is empty (default):
 *     /usr/share/models/person_detect.dla
 *     → /usr/share/models/person_detect.dla.onnx
 *
 * Always appends suffix. Always fails clearly if not found.
 */

#include "model_resolver.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>

int neuron_shim_resolve_model(const char* original_path,
                               const char* suffix,
                               const char* model_dir,
                               char* resolved,
                               size_t resolved_len) {
    if (!original_path || !suffix || !resolved) return -1;

    int written;

    if (model_dir && model_dir[0] != '\0') {
        /*
         * Redirect: take basename, put in model_dir
         *   /usr/share/models/person_detect.dla
         *   → /opt/models/person_detect.dla.onnx
         */
        char tmp[1024];
        snprintf(tmp, sizeof(tmp), "%s", original_path);
        const char* base = basename(tmp);

        /* Handle trailing slash on model_dir */
        size_t dir_len = strlen(model_dir);
        const char* sep = (dir_len > 0 && model_dir[dir_len - 1] == '/')
                          ? "" : "/";

        written = snprintf(resolved, resolved_len, "%s%s%s%s",
                           model_dir, sep, base, suffix);
    } else {
        /* Default: suffix in place */
        written = snprintf(resolved, resolved_len, "%s%s",
                           original_path, suffix);
    }

    if (written < 0 || (size_t)written >= resolved_len) {
        fprintf(stderr, "[neuron-shim] ERROR: resolved path too long\n");
        return -1;
    }

    /* Check file exists */
    if (access(resolved, R_OK) != 0) {
        fprintf(stderr,
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  neuron-shim: MODEL FILE NOT FOUND                      ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║                                                          ║\n"
            "║  Application requested:                                  ║\n"
            "║    %s\n"
            "║                                                          ║\n"
            "║  Shim looked for:                                        ║\n"
            "║    %s\n"
            "║                                                          ║\n",
            original_path, resolved);

        if (model_dir && model_dir[0] != '\0') {
            fprintf(stderr,
            "║  model_dir is set to:                                     ║\n"
            "║    %s\n"
            "║                                                          ║\n"
            "║  To fix, place your converted model there:               ║\n"
            "║    cp your_model%s %s\n",
                model_dir, suffix, resolved);
        } else {
            fprintf(stderr,
            "║  To fix, place the converted model next to the original: ║\n"
            "║    cp your_model%s %s\n"
            "║                                                          ║\n"
            "║  Or set model_dir in neuron-shim.conf to redirect:       ║\n"
            "║    model_dir = /opt/models                               ║\n",
                suffix, resolved);
        }

        fprintf(stderr,
            "║                                                          ║\n"
            "╚══════════════════════════════════════════════════════════╝\n\n");
        return -1;
    }

    return 0;
}
