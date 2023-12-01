/*
Copyright 2021 by Inria, Mickaël Ly, Jean Jouve, Florence Bertails-Descoubes and
    Laurence Boissieux

This file is part of ProjectiveFriction.

ProjectiveFriction is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

ProjectiveFriction is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
ProjectiveFriction. If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef CPP_UTILS_HPP
#define CPP_UTILS_HPP

/** \file
 *  Defines Types and procedure to ease the manipulation of the C++ standard
 *  library objects.
 */

#include <algorithm>
#include <functional>
#include <tuple>
#include <vector>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/tuple/tuple.hpp>

/**
 * @brief Stores the index and the value of an indexable container's element.
 *        It aims at avoiding the use of inappropriate data structures 
 *        such as tuples.
 */
template<typename value_type>
struct IndexAndValue
{
    IndexAndValue(const boost::tuple<std::size_t, value_type>& other)
    {
        boost::tie(index, value) = other;
    }
    std::size_t index;
    value_type value;
};

/**
 * @brief Create an IndexAndValue object from a tuple.
 */
template<typename value_type>
IndexAndValue<value_type>
getIndeAndValueFromTuple(const boost::tuple<std::size_t, value_type>& other)
{
    return IndexAndValue<value_type>(other);
}

/**
 * Return lhs.value < rhs.value.
 */
//TODO: SFINAE the function away in case of an unorderable type.
//      Define operator < ?
template<typename value_type>
bool
isValueLess(const IndexAndValue<value_type>& lhs,
            const IndexAndValue<value_type>& rhs)
{
    return lhs.value < rhs.value;
}

/**
 * Returns a sorted vector of the elements of the input range.
 * @tparam value_type
 */
template<typename IteratorOnOrderable>
std::vector<typename std::iterator_traits<IteratorOnOrderable>::value_type>
getSortedVector(IteratorOnOrderable first, IteratorOnOrderable last)
{
    std::vector<typename std::iterator_traits<IteratorOnOrderable>::value_type>
      result(first, last);
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * Returns an IndexAndValue iterator on the elements of vector.
 * @see IndexAndValue
 */
template<typename value_type>
auto
getIndexAndValueIteratorBegin(const std::vector<value_type>& vector)
{
    return boost::make_transform_iterator(
      boost::make_zip_iterator(
        boost::make_tuple(boost::make_counting_iterator(0ul), vector.begin())),
      &getIndeAndValueFromTuple<value_type>);
}

/**
 * Returns a past the end IndexAndValue iterator on the elements of vector.
 * @see IndexAndValue
 */
template<typename value_type>
auto
getIndexAndValueIteratorEnd(const std::vector<value_type>& vector)
{
    return boost::make_transform_iterator(
      boost::make_zip_iterator(boost::make_tuple(
        boost::make_counting_iterator(vector.size()), vector.end())),
      &getIndeAndValueFromTuple<value_type>);
}

/**
 * Returns an IndexAndValue range on the elements of vector.
 */
template<typename value_type>
auto
getIndexAndValueRange(const std::vector<value_type>& vector)
{
    return boost::make_iterator_range(getIndexAndValueIteratorBegin(vector),
                                      getIndexAndValueIteratorEnd(vector));
}

/**
 * @brief Construct an iterator on the dereference of the values ranged over by
 *        iterator. For example,
 * \code{.cpp}
 *  std::vector<int*> v;
 *  //Filling v.
 *  auto it = getDereferenceIterator(v.begin());
 *  assert(*it == *(*v.begin())); // This assert never fails.
 * \endcode
 * This is useful when you work with a vector of pointer, or similar.
 */
template<typename IteratorOnDereferencable>
auto
getDereferenceIterator(IteratorOnDereferencable iterator)
{
    return boost::make_transform_iterator(
      iterator,
      [](typename std::iterator_traits<IteratorOnDereferencable>::reference
           value) { return *value; });
}

/**
 * @brief Alias for the return type of getDereferenceIterator.
 */
template<typename IteratorOnDereferencable>
using DereferenceIterator =
  decltype(getDereferenceIterator(std::declval<IteratorOnDereferencable>()));

/**
 * @brief Abstraction of static_cast as a function. Its aim is to allow the use
 * of static_cast in functional environment. For example, as an std::transform
 * argument.
 */
template<typename To, typename From>
constexpr To staticCastFunction(From&& from)
{
    return static_cast<To>(std::forward<From>(from));
}

#endif // CPP_UTILS_HPP
