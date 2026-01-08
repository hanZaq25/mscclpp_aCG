/**
 * @file mscclpp_c_wrapper.cu
 * @brief C wrapper implementation for MSCCLPP's NCCL-compatible API
 * 
 * This file is compiled with nvcc (CUDA compiler) which supports C++.
 * It includes the real MSCCLPP header and forwards calls to the MSCCLPP library.
 * 
 * The wrapper-specific functions are implemented here, while standard NCCL
 * functions are provided by linking against libmscclpp_nccl.so.
 */

#include <mscclpp/nccl.h>
#include <stdio.h>
#include <stdlib.h>

/* Static state for the wrapper */
static int s_initialized = 0;
static int s_debug_level = 0;

/* Version string */
static const char* MSCCLPP_WRAPPER_VERSION = "1.0.0";

/*
 * ============================================================================
 * Debug/logging macros
 * ============================================================================
 */
#define WRAPPER_LOG(level, fmt, ...) \
    do { \
        if (s_debug_level >= level) { \
            fprintf(stderr, "[MSCCLPP_WRAPPER] " fmt "\n", ##__VA_ARGS__); \
        } \
    } while(0)

#define WRAPPER_ERROR(fmt, ...) WRAPPER_LOG(1, "ERROR: " fmt, ##__VA_ARGS__)
#define WRAPPER_WARN(fmt, ...)  WRAPPER_LOG(2, "WARN: " fmt, ##__VA_ARGS__)
#define WRAPPER_INFO(fmt, ...)  WRAPPER_LOG(3, "INFO: " fmt, ##__VA_ARGS__)
#define WRAPPER_DEBUG(fmt, ...) WRAPPER_LOG(4, "DEBUG: " fmt, ##__VA_ARGS__)

/*
 * ============================================================================
 * MSCCLPP wrapper-specific functions
 * ============================================================================
 */

extern "C" ncclResult_t mscclppWrapperInit(void) {
    if (s_initialized) {
        WRAPPER_WARN("mscclppWrapperInit called but already initialized");
        return ncclSuccess;
    }
    
    WRAPPER_INFO("Initializing MSCCLPP wrapper");
    
    /* Check for debug environment variable */
    const char* debug_env = getenv("MSCCLPP_WRAPPER_DEBUG");
    if (debug_env) {
        s_debug_level = atoi(debug_env);
        WRAPPER_INFO("Debug level set to %d", s_debug_level);
    }
    
    s_initialized = 1;
    WRAPPER_INFO("MSCCLPP wrapper initialized successfully");
    
    return ncclSuccess;
}

extern "C" ncclResult_t mscclppWrapperFinalize(void) {
    if (!s_initialized) {
        WRAPPER_WARN("mscclppWrapperFinalize called but not initialized");
        return ncclSuccess;
    }
    
    WRAPPER_INFO("Finalizing MSCCLPP wrapper");
    s_initialized = 0;
    
    return ncclSuccess;
}

extern "C" int mscclppWrapperIsInitialized(void) {
    return s_initialized;
}

extern "C" void mscclppWrapperSetDebugLevel(int level) {
    s_debug_level = level;
    WRAPPER_INFO("Debug level set to %d", level);
}

extern "C" const char* mscclppWrapperGetVersionString(void) {
    return MSCCLPP_WRAPPER_VERSION;
}

/*
 * ============================================================================
 * IMPORTANT NOTES
 * ============================================================================
 * 
 * 1. The mscclpp_c_wrapper.h header defines NCCL types in pure C for use by
 *    .c files. These definitions match MSCCLPP's definitions exactly.
 * 
 * 2. This .cu file includes <mscclpp/nccl.h> directly because nvcc can handle
 *    the C++ headers that MSCCLPP uses internally.
 * 
 * 3. Standard NCCL functions (ncclAllReduce, ncclCommInitRank, etc.) are
 *    implemented by MSCCLPP's library. You must link against:
 *      - libmscclpp_nccl.so (or mscclpp_nccl)
 * 
 * 4. The type definitions in mscclpp_c_wrapper.h are API-compatible with
 *    MSCCLPP's definitions, so you can safely pass ncclComm_t, ncclUniqueId,
 *    etc. between C and CUDA code.
 * ============================================================================
*/