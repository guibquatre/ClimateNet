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
#include "physics/CollisionConstraint.hpp"
#include "physics/PhysicScene.hpp"



CollisionConstraint::CollisionConstraint(
  std::size_t collision_vertex_index,
  const CollisionInfo& collision_info,
  double weight) :
  AbstractConstraint(std::vector<std::size_t>({ collision_vertex_index }),
                     weight),
  m_collision_info(collision_info)
{
}

CollisionConstraint::CollisionConstraint(
  std::size_t collision_vertex_index,
  CollisionInfo&& collision_info,
  double weight) :
  AbstractConstraint(std::vector<std::size_t>({ collision_vertex_index }),
                     weight),
  m_collision_info(std::move(collision_info))
{
}

void
CollisionConstraint::addConstraintMatrix(
  std::vector<SparseTriplet> &triplet_list)
  noexcept
{

  constraint_offset = 
    triplet_list.empty() ? 0u : (triplet_list.back().row() + 1u);

  triplet_list.push_back(
    SparseTriplet(constraint_offset, m_relevant_indices[0], m_weight));
  
}


void
CollisionConstraint::addConstraintProjection(
  const Eigen::VectorXd& positions,
  Eigen::Matrix3Xd& projections) const noexcept
{
  projections.col(constraint_offset) =
    //(1. / m_weight) *  // Fixing the fix of getProjection
    getProjection(positions, m_relevant_indices[0],
                  m_collision_info.contact_point,
                  m_collision_info.normal);
}



Eigen::Vector3d
CollisionConstraint::getProjection(const Eigen::VectorXd& positions,
                                   unsigned int vId,
                                   const Eigen::Vector3d& contact_pos,
                                   const Eigen::Vector3d& normal)
{
  
  const Eigen::Vector3d pos = positions.segment<3>(3u * vId);
  const Eigen::Vector3d point_to_state = pos - contact_pos;
  
  const double distance_to_plane = normal.dot(point_to_state);

  // Replace with epsilon
  const double eps = 5.0e-3;
  Eigen::Vector3d res =
    (distance_to_plane > eps) ?
    pos:
    (Eigen::Vector3d)(pos - (distance_to_plane - eps) * normal);


  return res;
}
