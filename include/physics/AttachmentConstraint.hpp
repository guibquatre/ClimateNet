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
#ifndef ATTACHMENT_CONSTRAINT_HPP
#define ATTACHMENT_CONSTRAINT_HPP

#include "physics/AbstractConstraint.hpp"

#include <Eigen/Core>

/**
 * @brief Constraint that add a spring between the constrained vertex and a 
 *        fixed position. It can be used to enforce a vertex to stay near a 
 *        position by giving the constraint a great weight compared to 
 *        other weights.
 */
class AttachmentConstraint : public AbstractConstraint
{

public:
  /**
   * @brief Construct an attachment constraint.
   * @param weight    The weight of the constraint
   * @param index     The index of the vertex to be constrained
   * @param position  The position to which the constrained vertex should
   *                  be attached
   */
  AttachmentConstraint(double weight,
                       std::size_t index,
                       const Eigen::Vector3d& position);

public:
  
  /**
   * @brief Request new lines in the sparse matrix containing the constraints
   *        (of size then 1 x nVertices)
   * @param triplet_list  Triplet list where to add the triplets (row, col, 0)
   */
  virtual void addConstraintMatrix(std::vector<SparseTriplet> &triplet_list)
    noexcept override final;


  /**
   * @brief Compute the B * projection (B_i p_i in the paper) and add it to the projections
   * @param positions    State to project (position or speed, depending on the constraint)
   * @param projections  The projection matrix
   */
  virtual void addConstraintProjection(
    const Eigen::VectorXd& positions,
    Eigen::Matrix3Xd& projections) const noexcept override final;
  
  /// @brief Attachment position
  Eigen::Vector3d m_position;
};

#endif // ATTACHEMENT_CONSTRAINT_HPP
