#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define TENSOR_MAX_DIMS   (12)

typedef int mkldnn_dims_t[TENSOR_MAX_DIMS];
typedef ptrdiff_t mkldnn_strides_t[TENSOR_MAX_DIMS];

/** Data type specification */
typedef enum {
    /** Undefined data type, used for empty memory descriptors. */
    mkldnn_data_type_undef = 0,
    /** 32-bit/single-precision floating point. */
    mkldnn_f32 = 1,
    /** 32-bit signed integer. */
    mkldnn_s32 = 2,
    /** 16-bit signed integer. */
    mkldnn_s16 = 4,
    /** 8-bit signed integer. */
    mkldnn_s8 = 5,
    /** 8-bit unsigned integer. */
    mkldnn_u8 = 6,
} mkldnn_data_type_t;


typedef struct {
  int ndims;
  mkldnn_dims_t dims;
  mkldnn_data_type_t data_type;

} mkldnn_memory_desc_t;

#ifdef __cplusplus
}
#endif

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

struct convolution_forward : public primitive {
  struct primitive_desc {
    // TODO(syoyo): Implement
    // src, weights, biasm dst
  };
};

struct pooling_forward : public primitive {

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



  struct desc {
    friend struct memory;

    mkldnn_memory_desc_t data;

    desc(dims adims, data_type adata_type, format aformat) {
      // TODO(syoyo): Implement
    }

  };

  struct primitive_desc {
    friend struct memory;

    // TODO: make private
    primitive_desc() :
      desc_({0}, data_type::f32, format::any) {
    }

    primitive_desc(const desc &adesc, const engine &aengine) : desc_(adesc) {
      //mkldnn_primitive_desc_t result;
      //error::wrap_c_api(
      //        mkldnn_memory_primitive_desc_create(&result,
      //            &adesc.data, aengine.get()),
      //        "could not initialize a memory primitive descriptor");
      //reset(result);

      // TODO(syoyo):
    }


    memory::desc desc() {
      return desc_;
    }

    struct desc desc_;
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

  void *get_data_handle() {
    return reinterpret_cast<void *>(data.data());
  }

 private:
  //int dim[4] = {0, 0, 0, 0};
  std::vector<float> data;
  primitive_desc primitive_desc_;

};
 


} // namespace
