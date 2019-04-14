// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "node.h"

namespace oidn {

#if OIDN_USE_NNPACK
  // Reorders weights from oihw to padded oihw format
  template<int K>
  class WeightsReorderNode : public Node
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    WeightsReorderNode(const std::shared_ptr<memory>& _src,
                       const std::shared_ptr<memory>& _dst)
      : src(_src),
        dst(_dst)
    {
      memory::primitive_desc srcDesc = src->get_primitive_desc();
      memory::primitive_desc dstDesc = dst->get_primitive_desc();
      MAYBE_UNUSED(srcDesc);
      MAYBE_UNUSED(dstDesc);
      assert(srcDesc.get_format() == memory::format::oihw);
      assert(dstDesc.get_format() == memory::format::oihw);
      assert(srcDesc.get_ndims() == 4);
      assert(dstDesc.get_ndims() == 4);
      assert(srcDesc.get_data_type() == memory::data_type::f32);
      assert(dstDesc.get_data_type() == memory::data_type::f32);
      assert(getPadded<K>(srcDesc.get_dims()[0]) == dstDesc.get_dims()[0]); // OC
      assert(getPadded<K>(srcDesc.get_dims()[1]) == dstDesc.get_dims()[1]); // IC
      assert(srcDesc.get_dims()[2] == dstDesc.get_dims()[2]);
      assert(srcDesc.get_dims()[3] == dstDesc.get_dims()[3]);
    }

    void execute() override
    {
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();

      const float* srcPtr = src->get_data();
      float* dstPtr = dst->get_data();

      const int OC1 = srcPrimDesc.get_dims()[0];
      const int OC2 = dstPrimDesc.get_dims()[0];
      const int IC1 = srcPrimDesc.get_dims()[1];
      const int IC2 = dstPrimDesc.get_dims()[1];
      const int H   = dstPrimDesc.get_dims()[2];
      const int W   = dstPrimDesc.get_dims()[3];

      for (int oc = 0; oc < OC2; ++oc)
      {
        for (int ic = 0; ic < IC2; ++ic)
        {
          for (int h = 0; h < H; ++h)
          {
            for (int w = 0; w < W; ++w)
            {
              // Output is in oihw format
              float* dstPtr_c = dstPtr + oc*IC2*H*W + ic*H*W + h*W + w;

              if (oc < OC1 && ic < IC1)
              {
                // Input is in oihw format
                const float* srcPtr_c = srcPtr + oc*IC1*H*W + ic*H*W + h*W + w;
                *dstPtr_c = *srcPtr_c;
              }
              else
              {
                // padding
                *dstPtr_c = 0;
              }
            }
          }
        }
      }
    }

    std::shared_ptr<std::vector<float>> getDst() const override { return dst; }
  };

#else

  // Reorders weights from oihw to padded oihw format
  template<int K>
  class WeightsReorderNode : public Node
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    WeightsReorderNode(const std::shared_ptr<memory>& src,
                       const std::shared_ptr<memory>& dst)
      : src(src),
        dst(dst)
    {
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;
      MAYBE_UNUSED(srcDesc);
      MAYBE_UNUSED(dstDesc);
      assert(srcDesc.format == memory::format::oihw);
      assert(dstDesc.format == memory::format::oihw);
      assert(srcDesc.ndims == 4);
      assert(dstDesc.ndims == 4);
      assert(srcDesc.data_type == memory::data_type::f32);
      assert(dstDesc.data_type == memory::data_type::f32);
      assert(getPadded<K>(srcDesc.dims[0]) == dstDesc.dims[0]); // OC
      assert(getPadded<K>(srcDesc.dims[1]) == dstDesc.dims[1]); // IC
      assert(srcDesc.dims[2] == dstDesc.dims[2]);
      assert(srcDesc.dims[3] == dstDesc.dims[3]);
    }

    void execute() override
    {
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;

      const float* srcPtr = (float*)src->get_data_handle();
      float* dstPtr = (float*)dst->get_data_handle();

      const int OC1 = srcDesc.dims[0];
      const int OC2 = dstDesc.dims[0];
      const int IC1 = srcDesc.dims[1];
      const int IC2 = dstDesc.dims[1];
      const int H   = dstDesc.dims[2];
      const int W   = dstDesc.dims[3];

      for (int oc = 0; oc < OC2; ++oc)
      {
        for (int ic = 0; ic < IC2; ++ic)
        {
          for (int h = 0; h < H; ++h)
          {
            for (int w = 0; w < W; ++w)
            {
              // Output is in oihw format
              float* dstPtr_c = dstPtr + oc*IC2*H*W + ic*H*W + h*W + w;

              if (oc < OC1 && ic < IC1)
              {
                // Input is in oihw format
                const float* srcPtr_c = srcPtr + oc*IC1*H*W + ic*H*W + h*W + w;
                *dstPtr_c = *srcPtr_c;
              }
              else
              {
                // padding
                *dstPtr_c = 0;
              }
            }
          }
        }
      }
    }

    std::shared_ptr<memory> getDst() const override { return dst; }
  };
#endif

} // namespace oidn
