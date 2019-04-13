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

  // 2x2 nearest-neighbor upsampling node
  template<int K>
  class UpsampleNode : public Node
  {
  private:
    std::shared_ptr<std::vector<float>> src;
    std::shared_ptr<std::vector<float>> dst;

  public:
    UpsampleNode(const std::shared_ptr<std::vector<float>>& src,
                 const std::shared_ptr<std::vector<float>>& dst)
      : src(src),
        dst(dst)
    {
      //memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      //memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      //const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      //const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;
      //MAYBE_UNUSED(srcDesc);
      //MAYBE_UNUSED(dstDesc);
      //assert(srcDesc.format == BlockedFormat<K>::nChwKc);
      //assert(dstDesc.format == BlockedFormat<K>::nChwKc);
      //assert(srcDesc.ndims == 4);
      //assert(dstDesc.ndims == 4);
      //assert(srcDesc.data_type == memory::data_type::f32);
      //assert(dstDesc.data_type == memory::data_type::f32);
      //assert(srcDesc.dims[0] == 1);
      //assert(dstDesc.dims[0] == 1);
      //// 2x2 upsampling
      //assert(dstDesc.dims[2] == srcDesc.dims[2] * 2);
      //assert(dstDesc.dims[3] == srcDesc.dims[3] * 2);
    }

    void execute() override
    {
      //memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      //const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;

      const float* srcPtr = src.get().data();
      float* dstPtr = dst.get().data();

      // TODO(LTE):
      const int C = 0; // srcDesc.dims[1];
      const int H = 0; // srcDesc.dims[2];
      const int W = 0; // srcDesc.dims[3];
      const int CK = C / K;

      //parallel_nd(CK, H, [&](int ck, int h)
      // TODO(syoyo): Parallel
      for (int h = 0; h < H; h++)
      {
        for (int ck = 0; ck < CK; ck++)
        {
          const size_t offset = ck*H*W*K + h*W*K;
          const float* srcPtr_line = srcPtr + offset;
          float* dstPtr_line0 = dstPtr + offset * 4;
          float* dstPtr_line1 = dstPtr_line0 + W*2*K; // next line

          for (int w = 0; w < W; ++w)
          {
            #pragma unroll
            for (int k = 0; k < K; k += 4)
            {
              const float *src_ptr = &srcPtr_line[w*K + k];
              memcpy(&dstPtr_line0[w*2*K   + k], src_ptr, sizeof(float) * 4);
              memcpy(&dstPtr_line0[w*2*K+K + k], src_ptr, sizeof(float) * 4);
              memcpy(&dstPtr_line1[w*2*K   + k], src_ptr, sizeof(float) * 4);
              memcpy(&dstPtr_line1[w*2*K+K + k], src_ptr, sizeof(float) * 4);
            }
          }
        }
      }
    }

    std::shared_ptr<std::vector<float>> getDst() const override { return dst; }
  };

#else

  // 2x2 nearest-neighbor upsampling node
  template<int K>
  class UpsampleNode : public Node
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    UpsampleNode(const std::shared_ptr<memory>& src,
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
      assert(srcDesc.format == BlockedFormat<K>::nChwKc);
      assert(dstDesc.format == BlockedFormat<K>::nChwKc);
      assert(srcDesc.ndims == 4);
      assert(dstDesc.ndims == 4);
      assert(srcDesc.data_type == memory::data_type::f32);
      assert(dstDesc.data_type == memory::data_type::f32);
      assert(srcDesc.dims[0] == 1);
      assert(dstDesc.dims[0] == 1);
      // 2x2 upsampling
      assert(dstDesc.dims[2] == srcDesc.dims[2] * 2);
      assert(dstDesc.dims[3] == srcDesc.dims[3] * 2);
    }

    void execute() override
    {
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;

      const float* srcPtr = (float*)src->get_data_handle();
      float* dstPtr = (float*)dst->get_data_handle();

      const int C = srcDesc.dims[1];
      const int H = srcDesc.dims[2];
      const int W = srcDesc.dims[3];
      const int CK = C / K;

      //parallel_nd(CK, H, [&](int ck, int h)
      // TODO(syoyo): Parallel
      for (int h = 0; h < H; h++)
      {
        for (int ck = 0; ck < CK; ck++)
        {
          const size_t offset = ck*H*W*K + h*W*K;
          const float* srcPtr_line = srcPtr + offset;
          float* dstPtr_line0 = dstPtr + offset * 4;
          float* dstPtr_line1 = dstPtr_line0 + W*2*K; // next line

          for (int w = 0; w < W; ++w)
          {
            #pragma unroll
            for (int k = 0; k < K; k += 4)
            {
#if 0
              const float *src_ptr = &srcPtr_line[w*K + k];
              memcpy(&dstPtr_line0[w*2*K   + k], src_ptr, sizeof(float) * 4);
              memcpy(&dstPtr_line0[w*2*K+K + k], src_ptr, sizeof(float) * 4);
              memcpy(&dstPtr_line1[w*2*K   + k], src_ptr, sizeof(float) * 4);
              memcpy(&dstPtr_line1[w*2*K+K + k], src_ptr, sizeof(float) * 4);
#else
              const __m128 m = _mm_load_ps(&srcPtr_line[w*K + k]);
              _mm_stream_ps(&dstPtr_line0[w*2*K   + k], m);
              _mm_stream_ps(&dstPtr_line0[w*2*K+K + k], m);
              _mm_stream_ps(&dstPtr_line1[w*2*K   + k], m);
              _mm_stream_ps(&dstPtr_line1[w*2*K+K + k], m);
#endif
            }
          }
        }
      }
    }

    std::shared_ptr<memory> getDst() const override { return dst; }
  };
#endif

} // namespace oidn
