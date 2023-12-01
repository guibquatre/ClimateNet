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
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "types.hpp"


/**
 * @brief General structur for a constraint
 */
class AbstractConstraint
{
public:

  /**
   * @brief Contruct an AbstractConstraint from relevant vertex indices and a
   *        weight.
   *
   * @param relevant_indices  The indices of the verticesw the constraint. 
   */
  AbstractConstraint(const std::vector<std::size_t>& relevant_indices,
                     double weight);

  /**
   * @brief Copy constructor
   * @param other
   */
  AbstractConstraint(const AbstractConstraint& other) = default;

  virtual ~AbstractConstraint() = default;

  /**
   * @brief Extract from a vector (typically of all the Dofs) the components
   *        useful for the constraint (aka the relevant blocks)
   * @param vector
   * @return Relevant subvector
   */
  Eigen::VectorXd getRelevantVector(const Eigen::VectorXd& vector) const;

  /** 
   * @brief Getter
   * @return The size of the vectors involved by the constraint
   */
  std::size_t getRelevantSize() const; 
public:
  /**
   * @brief Request new lines in the sparse matrix containing the constraints
   *        (of size then ? x nVertices)
   * @param triplet_list  Triplet list where to add the triplets (row, col, 0)
   */
  virtual void addConstraintMatrix(std::vector<SparseTriplet> &triplet_list)
    noexcept = 0;

  
  /**
   * @brief Compute the B * projection (B_i p_i in the paper) and add it to the projections
   * @param state        State to project (position or speed, depending on the constraint)
   * @param projections  The projection matrix
   */
  virtual void addConstraintProjection(
    const Eigen::VectorXd& state,
    Eigen::Matrix3Xd& projections) const noexcept = 0;

  size_t constraint_offset;

protected:
  /// @brief Weight of the constraint
  double m_weight;
  /// @brief Vertices indices of the constraint
  std::vector<std::size_t> m_relevant_indices;
};

#endif
