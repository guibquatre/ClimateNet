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
#ifndef UTIL_HPP
#define UTIL_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <boost/iterator/transform_iterator.hpp>
#include <numeric>

#include "types.hpp"

/** @file
 * Some utility functions to manipulated Eigen's vectors and matrices.
 */


// TODO: Find a way to put the templated functions in a descending order
// of abstraction. (getVectorsTotalSize should be under
// getVectorConcatenation).

/**
 * Returns the sum of the size of the vector over which the given range
 * iterates. This essentialy gives the size of the vector obtained by
 * concatenating the vector iterated over by the given range.
 * @tparam VectorIterator Iterator type iterating over objects that have a
 *                        `size` method
 */
template<typename VectorIterator>
std::size_t
getVectorsTotalSize(VectorIterator begin, VectorIterator end)
{
    using VectorType = typename std::iterator_traits<VectorIterator>::value_type;

    return std::accumulate(
      boost::make_transform_iterator(begin, std::bind(&VectorType::size, std::placeholders::_1)),
      boost::make_transform_iterator(end, std::bind(&VectorType::size, std::placeholders::_1)),
      0u);
}

/**
 * Returns the concatenation of the vectors iterated over by the given range.
 * The order of the vector is preserved in the concatenation.
 * @tparam VectorMultiPassIterator [Multi pass
 *                                 iterator](https://www.boost.org/doc/libs/1_71_0/libs/utility/MultiPassInputIterator.html)
 *                                 over eigen row vectors. The multi pass
 *                                 requirement is essential!
 */
template<typename VectorMultiPassIterator>
Eigen::VectorXd
getVectorConcatenation(VectorMultiPassIterator begin, VectorMultiPassIterator end)
{
    using VectorType = typename std::iterator_traits<VectorMultiPassIterator>::value_type;

    std::size_t result_size = getVectorsTotalSize(begin, end);

    Eigen::VectorXd result(result_size);
    std::size_t current_index = 0;
    while (begin != end)
    {
        // TODO: Use move affection when it is possible. Could be done
        // using C++17 constexpr if, or some temple black magic.
        result.segment(current_index, begin->size()) = *begin;
        current_index += begin->size();
        ++begin;
    }
    return result;
}

/**
 * Returns a reference to the 3x3 block whose top left corner is at the given
 * indices.
 * @param matrix The matrix from which the block is referenced.
 * @param row The index within the matrix of the top row of the referenced
 *            block.
 * @param colum The index within the matrix of the left most column of the
 *              referenced block.
 */
Eigen::Block<Matrix, 3, 3>
getMatrix3dBlock(Matrix& matrix, int row, int column);

/**
 * Returns a reference to the 3x3 block whose top left corner is at the given
 * indices.
 * @param matrix The matrix from which the block is referenced.
 * @param row The index within the matrix of the top row of the referenced
 *            block.
 * @param colum The index within the matrix of the left most column of the
 *              referenced block.
 */
const Eigen::Block<const Matrix, 3, 3>
getMatrix3dBlock(const Matrix& matrix, int row, int column);

/**
 * Returns a reference to the segment whose first element is at the given
 * index within the given vector.
 * @param vector The vector from which the segment is referenced.
 * @param index The index within the matrix of the first element of the
 *              referenced segment.
 */
Eigen::VectorBlock<Eigen::VectorXd, 3>
getVector3dBlock(Eigen::VectorXd& vector, int index);

/**
 * Returns a reference to the segment whose first element is at the given
 * index within the given vector.
 * @param vector The vector from which the segment is referenced.
 * @param index The index within the matrix of the first element of the
 *              referenced segment.
 */
const Eigen::VectorBlock<const Eigen::VectorXd, 3>
getVector3dBlock(const Eigen::VectorXd& vector, int index);

/**
 * Sets the given sparse matrix to 0.
 */
void
resetSparseMatrix(SparseMatrix& matrix);

/**
 * Return the number of 3d vector stored in the given vector. Meaning, its size
 * divided by 3.
 */
std::size_t
getVector3dSize(const Eigen::VectorXd& vector);

/**
 * Returns a 3`n`x3`n` vector when `n` is the given number.
 */
Eigen::MatrixXd
getMatrixOn3dVector(std::size_t nb_vector);

#endif
