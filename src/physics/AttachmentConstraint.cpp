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
#include "physics/AttachmentConstraint.hpp"


AttachmentConstraint::AttachmentConstraint(double weight,
                                           std::size_t index,
                                           const Eigen::Vector3d& position) :
  AbstractConstraint(std::vector<std::size_t>({ index }), weight),
  m_position(position)
{
}

void
AttachmentConstraint::addConstraintMatrix(
  std::vector<SparseTriplet> &triplet_list)
  noexcept
{
  constraint_offset =
    triplet_list.empty()?
    0u : (triplet_list.back().row() + 1u);

  triplet_list.push_back(
    SparseTriplet(constraint_offset, m_relevant_indices[0], m_weight));
}


void
AttachmentConstraint::addConstraintProjection(
  const Eigen::VectorXd& positions,
  Eigen::Matrix3Xd& projections) const noexcept
{
  projections.col(constraint_offset) = m_weight * m_position;
}
