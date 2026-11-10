// SPDX-License-Identifier: Apache-2.0
// ONNX Runtime custom op library that forwards selected ops to an externally
// registered callback (the NPU runtime). Zero-copy: the callback receives raw
// input/output tensor pointers.
//
// Domain "kokoro.npu", op "NpuOp":
//   input  X : float [1, C, T]
//   output Y : float [1, cout or C, T + t_delta]
//   attrs    : op_id (int64, opaque id passed to the callback)
//              cout  (int64, 0 = same as input channels)
//              t_delta (int64, output length delta)
//
// The host registers the callback with `npu_ops_set_callback` (via ctypes).
// Callback contract: return 0 on success; non-zero aborts the Run with an error.

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

extern "C" {
typedef int (*npu_op_callback_t)(int64_t op_id, const float* x, int64_t c, int64_t t, float* y,
                                 int64_t cout, int64_t tout);
}

namespace {

npu_op_callback_t g_callback = nullptr;
std::string g_last_error;

struct NpuOpKernel {
  NpuOpKernel(const OrtApi& api, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo ki(info);
    op_id_ = ki.GetAttribute<int64_t>("op_id");
    cout_ = ki.GetAttribute<int64_t>("cout");
    t_delta_ = ki.GetAttribute<int64_t>("t_delta");
  }

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    Ort::ConstValue x = ctx.GetInput(0);
    auto xi = x.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = xi.GetShape();
    if (shape.size() != 3 || shape[0] != 1) {
      ORT_CXX_API_THROW("NpuOp: expected input shape [1, C, T]", ORT_INVALID_ARGUMENT);
    }
    const int64_t c = shape[1];
    const int64_t t = shape[2];
    const int64_t cout = cout_ > 0 ? cout_ : c;
    const int64_t tout = t + t_delta_;
    if (tout < 0) ORT_CXX_API_THROW("NpuOp: negative output length", ORT_INVALID_ARGUMENT);
    std::vector<int64_t> out_shape{1, cout, tout};
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    if (!g_callback) ORT_CXX_API_THROW("NpuOp: no callback registered", ORT_FAIL);
    int rc = g_callback(op_id_, x.GetTensorData<float>(), c, t, y.GetTensorMutableData<float>(), cout, tout);
    if (rc != 0) {
      std::string msg = "NpuOp callback failed (op_id=" + std::to_string(op_id_) + ", rc=" + std::to_string(rc) + ")";
      ORT_CXX_API_THROW(msg.c_str(), ORT_FAIL);
    }
  }

  int64_t op_id_ = 0;
  int64_t cout_ = 0;
  int64_t t_delta_ = 0;
};

struct NpuOp : Ort::CustomOpBase<NpuOp, NpuOpKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new NpuOpKernel(api, info); }
  const char* GetName() const { return "NpuOp"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
};

const NpuOp c_NpuOp;
std::once_flag g_domain_once;
Ort::CustomOpDomain* g_domain = nullptr;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) void npu_ops_set_callback(npu_op_callback_t cb) { g_callback = cb; }

__attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                                                 const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  try {
    std::call_once(g_domain_once, []() {
      g_domain = new Ort::CustomOpDomain("kokoro.npu");
      g_domain->Add(&c_NpuOp);
    });
    Ort::UnownedSessionOptions so(options);
    so.Add(*g_domain);
  } catch (const Ort::Exception& e) {
    return Ort::GetApi().CreateStatus(e.GetOrtErrorCode(), e.what());
  } catch (const std::exception& e) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, e.what());
  }
  return nullptr;
}
}
