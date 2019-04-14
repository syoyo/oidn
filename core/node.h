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

#include "common.h"
#include <vector>

namespace oidn {

  class Node
  {
  public:
    virtual ~Node() = default;
    virtual void execute() = 0;
    virtual std::shared_ptr<memory> getDst() const { return nullptr; }
  };

#if defined(OIDN_USE_NNPACK)
  // Node wrapping an NNPACK primitive
  class NnpNode : public Node
  {
  private:
    //std::vector<primitive> net;

  public:
    NnpNode()
    {
    }

    void execute() override
    {
      //stream(stream::kind::eager).submit(net).wait();
    }
  };
#else
  // Node wrapping an MKL-DNN primitive
  class MklNode : public Node
  {
  private:
    std::vector<primitive> net;

  public:
    MklNode(const primitive& prim)
    {
      net.push_back(prim);
    }

    void execute() override
    {
      stream(stream::kind::eager).submit(net).wait();
    }
  };
#endif

#if defined(OIDN_USE_NNPACK)
  // Convolution node
  class ConvNode : public NnpNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> bias;
    std::shared_ptr<memory> dst;

  public:
    ConvNode(const memory::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& weights,
             const std::shared_ptr<memory>& bias,
             const std::shared_ptr<memory>& dst)
      : src(src), weights(weights), bias(bias), dst(dst) {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Pooling node
  class PoolNode : public NnpNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    // TODO(syoyo): Desc
    PoolNode(const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& dst)
      : src(src), dst(dst) {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };
#else
  // Convolution node
  class ConvNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> bias;
    std::shared_ptr<memory> dst;

  public:
    ConvNode(const convolution_forward::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& weights,
             const std::shared_ptr<memory>& bias,
             const std::shared_ptr<memory>& dst)
      : MklNode(convolution_forward(desc, *src, *weights, *bias, *dst)),
        src(src), weights(weights), bias(bias), dst(dst) {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Pooling node
  class PoolNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    PoolNode(const pooling_forward::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& dst)
      : MklNode(pooling_forward(desc, *src, *dst)),
        src(src), dst(dst) {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };
#endif

} // namespace oidn
