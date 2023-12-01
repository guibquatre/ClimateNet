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
#include <iostream>
#include <vector>

#include "cpp_utils.hpp"
#include "util.hpp"

#include "physics/AbstractConstraint.hpp"

AbstractConstraint::AbstractConstraint(
  const std::vector<std::size_t>& relevant_indices,
  double weight) :
  m_weight(sqrt(weight)),
  m_relevant_indices(relevant_indices)

{
}


Eigen::VectorXd
AbstractConstraint::getRelevantVector(const Eigen::VectorXd& vector) const
{
    Eigen::VectorXd relevant_vector(getRelevantSize());
    for (std::size_t i = 0; i < m_relevant_indices.size(); ++i)
    {
        getVector3dBlock(relevant_vector, i) =
          getVector3dBlock(vector, m_relevant_indices[i]);
    }
    return relevant_vector;
}

std::size_t
AbstractConstraint::getRelevantSize() const
{
    return m_relevant_indices.size() * 3;
}
