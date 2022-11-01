// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

/**
 * @defgroup ov_element_cpp_api Element types
 * @ingroup ov_model_cpp_api
 * OpenVINO Element API to work with OpenVINO element types
 *
 */

namespace ov {
namespace element {
/// \brief Enum to define possible element types
/// \ingroup ov_element_cpp_api
enum class Type_t {
    undefined,  //!< Undefined element type
    dynamic,    //!< Dynamic element type
    boolean,    //!< boolean element type
    bf16,       //!< bf16 element type
    f16,        //!< f16 element type
    f32,        //!< f32 element type
    f64,        //!< f64 element type
    i4,         //!< i4 element type
    i8,         //!< i8 element type
    i16,        //!< i16 element type
    i32,        //!< i32 element type
    i64,        //!< i64 element type
    u1,         //!< binary element type
    u4,         //!< u4 element type
    u8,         //!< u8 element type
    u16,        //!< u16 element type
    u32,        //!< u32 element type
    u64        //!< u64 element type
};

/// \brief Base class to define element type
/// \ingroup ov_element_cpp_api
class OPENVINO_API Type {
public:
    Type() = default;
    Type(const Type&) = default;
    constexpr Type(const Type_t t) : m_type{t} {}
    Type(size_t bitwidth, bool is_real, bool is_signed, bool is_quantized, const std::string& cname);
    Type& operator=(const Type&) = default;
    std::string c_type_string() const;
    size_t size() const;
    size_t hash() const;
    bool is_static() const;
    bool is_dynamic() const {
        return !is_static();
    }
    bool is_real() const;
    // TODO: We may want to revisit this definition when we do a more general cleanup of
    // element types:
    bool is_integral() const {
        return !is_real();
    }
    bool is_integral_number() const;
    bool is_signed() const;
    bool is_quantized() const;
    size_t bitwidth() const;
    // The name of this type, the enum name of this type
    std::string get_type_name() const;
    friend OPENVINO_API std::ostream& operator<<(std::ostream&, const Type&);
    static std::vector<const Type*> get_known_types();

    /// \brief Checks whether this element type is merge-compatible with `t`.
    /// \param t The element type to compare this element type to.
    /// \return `true` if this element type is compatible with `t`, else `false`.
    bool compatible(const element::Type& t) const;

    /// \brief Merges two element types t1 and t2, writing the result into dst and
    ///        returning true if successful, else returning false.
    ///
    ///        To "merge" two element types t1 and t2 is to find the least restrictive
    ///        element type t that is no more restrictive than t1 and t2, if t exists.
    ///        More simply:
    ///
    ///           merge(dst,element::Type::dynamic,t)
    ///              writes t to dst and returns true
    ///
    ///           merge(dst,t,element::Type::dynamic)
    ///              writes t to dst and returns true
    ///
    ///           merge(dst,t1,t2) where t1, t2 both static and equal
    ///              writes t1 to dst and returns true
    ///
    ///           merge(dst,t1,t2) where t1, t2 both static and unequal
    ///              does nothing to dst, and returns false
    static bool merge(element::Type& dst, const element::Type& t1, const element::Type& t2);

    // \brief This allows switch(element_type)
    constexpr operator Type_t() const {
        return m_type;
    }

private:
    Type_t m_type{Type_t::undefined};
};

using TypeVector = std::vector<Type>;

/// \brief undefined element type
/// \ingroup ov_element_cpp_api
constexpr Type undefined(Type_t::undefined);
/// \brief dynamic element type
/// \ingroup ov_element_cpp_api
constexpr Type dynamic(Type_t::dynamic);
/// \brief boolean element type
/// \ingroup ov_element_cpp_api
constexpr Type boolean(Type_t::boolean);
/// \brief bf16 element type
/// \ingroup ov_element_cpp_api
constexpr Type bf16(Type_t::bf16);
/// \brief f16 element type
/// \ingroup ov_element_cpp_api
constexpr Type f16(Type_t::f16);
/// \brief f32 element type
/// \ingroup ov_element_cpp_api
constexpr Type f32(Type_t::f32);
/// \brief f64 element type
/// \ingroup ov_element_cpp_api
constexpr Type f64(Type_t::f64);
/// \brief i4 element type
/// \ingroup ov_element_cpp_api
constexpr Type i4(Type_t::i4);
/// \brief i8 element type
/// \ingroup ov_element_cpp_api
constexpr Type i8(Type_t::i8);
/// \brief i16 element type
/// \ingroup ov_element_cpp_api
constexpr Type i16(Type_t::i16);
/// \brief i32 element type
/// \ingroup ov_element_cpp_api
constexpr Type i32(Type_t::i32);
/// \brief i64 element type
/// \ingroup ov_element_cpp_api
constexpr Type i64(Type_t::i64);
/// \brief binary element type
/// \ingroup ov_element_cpp_api
constexpr Type u1(Type_t::u1);
/// \brief u4 element type
/// \ingroup ov_element_cpp_api
constexpr Type u4(Type_t::u4);
/// \brief u8 element type
/// \ingroup ov_element_cpp_api
constexpr Type u8(Type_t::u8);
/// \brief u16 element type
/// \ingroup ov_element_cpp_api
constexpr Type u16(Type_t::u16);
/// \brief u32 element type
/// \ingroup ov_element_cpp_api
constexpr Type u32(Type_t::u32);
/// \brief u64 element type
/// \ingroup ov_element_cpp_api
constexpr Type u64(Type_t::u64);

template <typename T>
Type from() {
    throw std::invalid_argument("Unknown type");
}
template <>
OPENVINO_API Type from<char>();
template <>
OPENVINO_API Type from<bool>();
template <>
OPENVINO_API Type from<float>();
template <>
OPENVINO_API Type from<double>();
template <>
OPENVINO_API Type from<int8_t>();
template <>
OPENVINO_API Type from<int16_t>();
template <>
OPENVINO_API Type from<int32_t>();
template <>
OPENVINO_API Type from<int64_t>();
template <>
OPENVINO_API Type from<uint8_t>();
template <>
OPENVINO_API Type from<uint16_t>();
template <>
OPENVINO_API Type from<uint32_t>();
template <>
OPENVINO_API Type from<uint64_t>();
template <>
OPENVINO_API Type from<ov::bfloat16>();
template <>
OPENVINO_API Type from<ov::float16>();

OPENVINO_API Type fundamental_type_for(const Type& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& out, const ov::element::Type& obj);

OPENVINO_API
std::istream& operator>>(std::istream& out, ov::element::Type& obj);
}  // namespace element

template <>
class OPENVINO_API AttributeAdapter<ov::element::Type_t> : public EnumAttributeAdapterBase<ov::element::Type_t> {
public:
    AttributeAdapter(ov::element::Type_t& value) : EnumAttributeAdapterBase<ov::element::Type_t>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::element::Type_t>");
    BWDCMP_RTTI_DECLARATION;
};

template <>
class OPENVINO_API AttributeAdapter<ov::element::Type> : public ValueAccessor<std::string> {
public:
    OPENVINO_RTTI("AttributeAdapter<ov::element::Type>");
    BWDCMP_RTTI_DECLARATION;
    AttributeAdapter(ov::element::Type& value) : m_ref(value) {}

    const std::string& get() override;
    void set(const std::string& value) override;

    operator ov::element::Type&() {
        return m_ref;
    }

protected:
    ov::element::Type& m_ref;
};

template <>
class OPENVINO_API AttributeAdapter<ov::element::TypeVector> : public DirectValueAccessor<ov::element::TypeVector> {
public:
    OPENVINO_RTTI("AttributeAdapter<ov::element::TypeVector>");
    BWDCMP_RTTI_DECLARATION;
    AttributeAdapter(ov::element::TypeVector& value) : DirectValueAccessor<ov::element::TypeVector>(value) {}
};

}  // namespace ov
