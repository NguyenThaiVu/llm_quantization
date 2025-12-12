// cutlass_ext/transform/scale_op.h
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace transform {
namespace threadblock {

struct IdentityOp {
  struct Params {};
  CUTLASS_HOST_DEVICE IdentityOp(Params const& = Params()) {}

  template <typename T>
  CUTLASS_HOST_DEVICE T operator()(T x) const { return x; }
};


struct ScaleOp {
  struct Params {
    float scale;
    CUTLASS_HOST_DEVICE Params(float s = 1.f) : scale(s) {}
  };

  float scale;
  CUTLASS_HOST_DEVICE ScaleOp(Params const& p = Params()) : scale(p.scale) {}

  CUTLASS_HOST_DEVICE
  cutlass::bfloat16_t operator()(cutlass::bfloat16_t x) const {
    cutlass::NumericConverter<float, cutlass::bfloat16_t> to_f;
    cutlass::NumericConverter<cutlass::bfloat16_t, float> to_bf16;
    return to_bf16(to_f(x) * scale);
  }
};
} // threadblock
} // transform
} // cutlass
