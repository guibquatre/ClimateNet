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
#ifndef COLLISION_CONSTRAINT_HPP
#define COLLISION_CONSTRAINT_HPP

#include <Eigen/Core>

#include "physics/AbstractConstraint.hpp"
#include "physics/Obstacle.hpp"



/// @brief Class implementing a collision by projection
class CollisionConstraint : public AbstractConstraint
{
  
public:
  /**
   * @brief Constructor
   * @param collision_vertex_index  Index of the vertex in collision
   * @param collision_info        Point and normal on the obstacle
   * @param weight                  Weight of the constraints
   */
  CollisionConstraint(std::size_t collision_vertex_index,
                      const CollisionInfo& collision_info,
                      double weight);
  /**
   * @brief Constructor
   * @param collision_vertex_index  Index of the vertex in collision
   * @param collision_info        Point and normal on the obstacle
   * @param weight                  Weight of the constraints
   */
  CollisionConstraint(std::size_t collision_vertex_index,
                      CollisionInfo&& collision_info,
                      double weight);

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

  
  static Eigen::Vector3d
  getProjection(const Eigen::VectorXd& state,
                unsigned int vId,
                const Eigen::Vector3d& contact_pos,
                const Eigen::Vector3d& normal);

private:
  /// @brief Contact information
  CollisionInfo m_collision_info;
};

#endif // COLLISION_CONSTRAINT_HPP
