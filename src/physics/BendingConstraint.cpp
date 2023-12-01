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
#include "physics/BendingConstraint.hpp"

#include <Eigen/Geometry>

#include "geometry/geometry_util.hpp"
#include "util.hpp"

BendingConstraint::BendingConstraint(const std::vector<std::size_t>& vertex_indices,
                                     const Eigen::Vector3d& middle_vertex_rest_state,
                                     const std::vector<Eigen::Vector3d>& rest_state,
                                     double weight)
  : BendingConstraint(std::vector<std::size_t>(vertex_indices),
                      middle_vertex_rest_state,
                      rest_state,
                      weight)
{
}
BendingConstraint::BendingConstraint(const std::vector<std::size_t>& vertex_indices,
                                     Eigen::Vector3d&& middle_vertex_rest_state,
                                     std::vector<Eigen::Vector3d>&& rest_state,
                                     double weight)
  : BendingConstraint(std::vector<std::size_t>(vertex_indices),
                      middle_vertex_rest_state,
                      rest_state,
                      weight)
{
}
BendingConstraint::BendingConstraint(std::vector<std::size_t>&& vertex_indices,
                                     const Eigen::Vector3d& middle_vertex_rest_state,
                                     const std::vector<Eigen::Vector3d>& rest_state,
                                     double weight)
  : AbstractConstraint(std::move(vertex_indices), weight)
{
    assert(rest_state.size() > 3);
    assert(getRelevantSize() / 3 == rest_state.size() + 1);

    // TODO: factorize with update()
    m_A = weight * getLaplaceBeltramiDicretisation(middle_vertex_rest_state, rest_state);
    m_ATA = m_A.transpose() * m_A;

    // This equality come from the definition of the discretisation of
    // the laplace beltrami operator.
    m_laplace_beltrami_coefficient_sum = -m_A(0, 0);

    Eigen::VectorXd generalized_rest_state((rest_state.size() + 1) * 3);
    for (std::size_t i = 0; i < rest_state.size(); ++i)
    {
        getVector3dBlock(generalized_rest_state, i + 1) = rest_state[i];
    }
    getVector3dBlock(generalized_rest_state, 0) = middle_vertex_rest_state;

    m_rest_state_laplace_beltrami_norm = applyA(generalized_rest_state).norm() ; //(m_A * generalized_rest_state).norm();

    const std::vector<double> coeffs =
      getLaplaceBeltramiDiscretisationMeanValueCoefficients(middle_vertex_rest_state, rest_state);
    m_beltrami_coefficients.reserve(m_relevant_indices.size());
    m_beltrami_coefficients.push_back(0.);
    for (size_t i = 0u; i < coeffs.size(); ++i)
    {
      m_beltrami_coefficients[0] += coeffs[i];
      m_beltrami_coefficients.push_back(-coeffs[i]);
    }
    
}

BendingConstraint::BendingConstraint(std::vector<std::size_t>&& vertex_indices,
                                     Eigen::Vector3d&& middle_vertex_rest_state,
                                     std::vector<Eigen::Vector3d>&& rest_state,
                                     double weight)
  : BendingConstraint(std::move(vertex_indices), middle_vertex_rest_state, rest_state, weight)
{
}

// A
void
BendingConstraint::addConstraintMatrix(std::vector<SparseTriplet>& triplet_list) noexcept
{

  constraint_offset = triplet_list.empty() ? 0u : (triplet_list.back().row() + 1u);
  for (size_t i = 0u; i < m_relevant_indices.size(); ++i)
  {
    triplet_list.push_back(SparseTriplet(
                             constraint_offset, m_relevant_indices[i], m_weight *  m_beltrami_coefficients[i]));
  } // i
}

void
BendingConstraint::addConstraintProjection(const Eigen::VectorXd& positions,
                                           Eigen::Matrix3Xd& projections) const noexcept
{

    Eigen::Vector3d laplace_beltrami = applyA(getRelevantVector(positions));

    double laplace_beltrami_norm = laplace_beltrami.norm();

    // If the laplace beltrami is the null vector we do not normalize it
    if (laplace_beltrami_norm > 1e-25)
    {
        laplace_beltrami *= m_rest_state_laplace_beltrami_norm / laplace_beltrami_norm;
        projections.col(constraint_offset) = m_weight * laplace_beltrami;
    }
    else
    {
      projections.col(constraint_offset).setZero();
    }
}

Eigen::Vector3d
BendingConstraint::applyA(const Eigen::VectorXd& relevant_vector) const noexcept
{
    Eigen::Vector3d result(Eigen::Vector3d::Zero());
    for (std::size_t vertex_index = 0; vertex_index < m_beltrami_coefficients.size();
         ++vertex_index)
    {
        result +=
          m_beltrami_coefficients[vertex_index] * getVector3dBlock(relevant_vector, vertex_index);
    }
    return result;
}
