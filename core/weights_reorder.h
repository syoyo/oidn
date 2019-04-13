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
    std::shared_ptr<std::vector<float>> src;
    std::shared_ptr<std::vector<float>> dst;

  public:
    WeightsReorderNode(const std::shared_ptr<std::vector<float>>& _src,
                       const std::shared_ptr<std::vector<float>>& _dst)
      : src(_src),
        dst(_dst)
    {
      //memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      //memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      //const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      //const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;
      //MAYBE_UNUSED(srcDesc);
      //MAYBE_UNUSED(dstDesc);
      //assert(srcDesc.format == memory::format::oihw);
      //assert(dstDesc.format == memory::format::oihw);
      //assert(srcDesc.ndims == 4);
      //assert(dstDesc.ndims == 4);
      //assert(srcDesc.data_type == memory::data_type::f32);
      //assert(dstDesc.data_type == memory::data_type::f32);
      //assert(getPadded<K>(srcDesc.dims[0]) == dstDesc.dims[0]); // OC
      //assert(getPadded<K>(srcDesc.dims[1]) == dstDesc.dims[1]); // IC
      //assert(srcDesc.dims[2] == dstDesc.dims[2]);
      //assert(srcDesc.dims[3] == dstDesc.dims[3]);
    }

    void execute() override
    {
      //memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      //memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      //const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      //const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;

      const float* srcPtr = src.get().data();
      float* dstPtr = dst.get().data();

      // TODO(LTE):
      const int OC1 = 0; //srcDesc.dims[0];
      const int OC2 = 0; //dstDesc.dims[0];
      const int IC1 = 0; //srcDesc.dims[1];
      const int IC2 = 0; //dstDesc.dims[1];
      const int H   = 0; //dstDesc.dims[2];
      const int W   = 0; //dstDesc.dims[3];

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
