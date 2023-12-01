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
#include <Eigen/Core>
#include <vector>
#include <iostream>

#include "util.hpp"

#include "physics/SpringConstraint.hpp"


const std::size_t SpringConstraint::m_nb_position_involved = 2;


SpringConstraint::SpringConstraint(std::size_t vertex1_index,
                                   std::size_t vertex2_index,
                                   double weight,
                                   double rest_length)
  : AbstractConstraint(std::vector<std::size_t>{ vertex1_index, vertex2_index },
                       weight) ,
    m_rest_length(rest_length)
{
}

void
SpringConstraint::addConstraintMatrix(
  std::vector<SparseTriplet> &triplet_list)
  noexcept
{
  constraint_offset = 
    triplet_list.empty() ? 0u : (triplet_list.back().row() + 1u);

  triplet_list.push_back(
    SparseTriplet(constraint_offset, m_relevant_indices[0], m_weight));
  triplet_list.push_back(
    SparseTriplet(constraint_offset, m_relevant_indices[1], -m_weight));
}


void
SpringConstraint::addConstraintProjection(
  const Eigen::VectorXd& positions,
  Eigen::Matrix3Xd& projections) const noexcept
{  
  Eigen::VectorXd e =
    positions.segment<3>(3u * m_relevant_indices[0])
    - positions.segment<3>(3u * m_relevant_indices[1]);
  
  e.normalize();  
  projections.col(constraint_offset) =
    m_weight * m_rest_length * e;
}
