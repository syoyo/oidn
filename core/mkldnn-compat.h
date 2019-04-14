#pragma once

namespace mkldnn_compat {

class primitive {
 public:


 private:

};


struct engine {

  /// Kinds of engines
  enum kind {
      /// An unspecified engine
      any,
      /// CPU engine
      cpu,
  };

  engine(kind akind, size_t index) {
    (void)akind;
    (void)index;
  }
};

struct memory : public primitive {

 public:
  typedef std::vector<int> dims;

  enum data_type {
   data_undef,
   f32
  };

  enum format {
   format_undef,
   any,
   blocked,
   x,
   nCw8c,
   nChw8c,
   nChw16c,
   oihw,
   OIhw8i8o,
   OIhw16i16o,
  };


  struct primitive_desc {
    friend struct memory;

    // TODO: make private
    primitive_desc() :
      data_type_(data_type::f32), format_(format::any) {
    }

    primitive_desc(const dims &dims, const memory::data_type data_type, const memory::format format) :
      dims_(dims), data_type_(data_type), format_(format) {
    }

    dims get_dims() const {
      return dims_;
    }

    int get_ndims() const {
      return int(dims_.size());
    }

    data_type get_data_type() const {
      return data_type_;
    }

    format get_format() const {
      return format_;
    }

    dims dims_;
    data_type data_type_;
    format format_;
  };

  memory(const primitive_desc &adesc) : primitive_desc_(adesc) {

  }

  memory(const primitive_desc &adesc, void *ahandle) : primitive_desc_(adesc) {
    (void)ahandle;
    // TODO(syoyo): handle
  }


  primitive_desc get_primitive_desc() const {
     return primitive_desc_;
  }

  const float *get_data() const {
    return data_.data();
  }

  float *get_data() {
    return data_.data();
  }

 private:
  struct primitive_desc primitive_desc_;
  std::vector<float> data_; // TODO(LTE): use uint8 for supporting arbitrary tensor type

};



} // namespace
