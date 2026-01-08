/**
 * @file mscclpp_c_wrapper.h
 * @brief C wrapper for MSCCLPP's NCCL-compatible API
 * 
 * This header provides a pure C interface to MSCCLPP's NCCL compatibility layer.
 * It can be included from both C (.c) and CUDA (.cu) files.
 * 
 * The NCCL types are defined here for C compatibility, matching the definitions
 * in <mscclpp/nccl.h>. The actual implementations are provided by MSCCLPP's
 * library and our wrapper .cu file.
 */

#ifndef MSCCLPP_C_WRAPPER_H
#define MSCCLPP_C_WRAPPER_H

#include <stddef.h>
#include <limits.h>

/* 
 * Handle cudaStream_t definition:
 * - If CUDA headers are already included, cudaStream_t is already defined
 * - If not, we define it as an opaque pointer for pure C compilation
 */
#ifndef __DRIVER_TYPES_H__
#ifndef __CUDA_RUNTIME_API_H__
#ifndef CUDA_RUNTIME_H
/* No CUDA headers included yet - define cudaStream_t as opaque pointer */
typedef struct CUstream_st *cudaStream_t;
#endif
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * NCCL Type Definitions (matching mscclpp/nccl.h)
 * ============================================================================
 * These must match exactly with MSCCLPP's definitions to ensure API compatibility.
 */

/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;
#define NCCL_COMM_NULL NULL

/* Unique ID for communicator initialization */
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Error type */
typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
} ncclResult_t;

/* Data types */
typedef enum {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes = 12
} ncclDataType_t;

/* Reduction operation selector */
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;
typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5,
    ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t))
} ncclRedOp_t;

/*
 * ============================================================================
 * NCCL API Functions
 * ============================================================================
 * These are implemented by MSCCLPP's library (libmscclpp_nccl.so)
 */

/* Version */
ncclResult_t ncclGetVersion(int* version);

/* Unique ID */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);

/* Communicator management */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t ncclCommFinalize(ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);

/* Error handling */
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);

/* Communicator queries */
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

/* Memory management */
ncclResult_t ncclMemAlloc(void** ptr, size_t size);
ncclResult_t ncclMemFree(void* ptr);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle);

/* Collective operations */
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/* Group operations */
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

/*
 * ============================================================================
 * MSCCLPP Wrapper-specific functions
 * ============================================================================
 */

/**
 * @brief Initialize MSCCLPP wrapper
 */
ncclResult_t mscclppWrapperInit(void);

/**
 * @brief Finalize MSCCLPP wrapper
 */
ncclResult_t mscclppWrapperFinalize(void);

/**
 * @brief Check if wrapper is initialized
 */
int mscclppWrapperIsInitialized(void);

/**
 * @brief Set debug level (0=none, 1=error, 2=warn, 3=info, 4=debug)
 */
void mscclppWrapperSetDebugLevel(int level);

/**
 * @brief Get wrapper version string
 */
const char* mscclppWrapperGetVersionString(void);

#ifdef __cplusplus
}
#endif

#endif /* MSCCLPP_C_WRAPPER_H */