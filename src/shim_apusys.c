/*
 * neuron-shim: libapusys stub
 *
 * libapusys.so is the low-level APU driver interface that talks to
 * /dev/apusys. The Neuron Runtime uses it internally. If any
 * application or library tries to use apusys directly, these stubs
 * prevent crashes.
 *
 * Most of these are ioctl wrappers. We just return success.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Session management                                                  */
/* ------------------------------------------------------------------ */

/* From the kernel log patterns: apusysSession creates a session */
int apusys_session_create(void** session, int flags) {
    static int fake_session;
    if (session) *session = &fake_session;
    fprintf(stderr, "[neuron-shim][apusys] session_create (stub)\n");
    return 0;
}

int apusys_session_destroy(void* session) {
    fprintf(stderr, "[neuron-shim][apusys] session_destroy (stub)\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Command / execution                                                 */
/* ------------------------------------------------------------------ */

int apusys_cmd_create(void* session, int type, void** cmd) {
    static int fake_cmd;
    if (cmd) *cmd = &fake_cmd;
    return 0;
}

int apusys_cmd_destroy(void* session, void* cmd) {
    return 0;
}

int apusys_cmd_run(void* cmd) {
    fprintf(stderr, "[neuron-shim][apusys] cmd_run (stub, no-op)\n");
    return 0;
}

int apusys_cmd_run_async(void* cmd) {
    return 0;
}

int apusys_cmd_wait(void* cmd, int timeout_ms) {
    return 0;
}

/* ------------------------------------------------------------------ */
/* Memory management                                                   */
/* ------------------------------------------------------------------ */

int apusys_mem_alloc(void* session, size_t size, void** mem) {
    /* Actually allocate — some code may write to this */
    if (mem) {
        *mem = calloc(1, size);
        if (!*mem) return -1;
    }
    return 0;
}

int apusys_mem_free(void* session, void* mem) {
    /* We can't safely free since we don't know if it was our alloc */
    return 0;
}

int apusys_mem_map(void* session, void* mem) {
    return 0;
}

int apusys_mem_unmap(void* session, void* mem) {
    return 0;
}

/* ------------------------------------------------------------------ */
/* Device info / power                                                 */
/* ------------------------------------------------------------------ */

int apusys_device_get_num(int type) {
    /* Report 1 device of each type */
    return 1;
}

int apusys_power_on(void* session) {
    return 0;
}

int apusys_power_off(void* session) {
    return 0;
}

/* ------------------------------------------------------------------ */
/* Firmware loading — the real driver loads APU firmware blobs          */
/* ------------------------------------------------------------------ */

int apusys_load_firmware(const char* path) {
    fprintf(stderr, "[neuron-shim][apusys] load_firmware: %s (ignored)\n",
            path ? path : "(null)");
    return 0;
}

#ifdef __cplusplus
}
#endif
