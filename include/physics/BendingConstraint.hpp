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
#ifndef BENDING_CONSTRAINT_HPP
#define BENDING_CONSTRAINT_HPP

#include "physics/AbstractConstraint.hpp"


/// @brief Class implementing a bend constraint around a vertex

class BendingConstraint : public AbstractConstraint
{
public:
  /**
   * @brief Construct a bending constraint from the indices of the 
   *        vertex involved in the constraint and from a rest state.
   *        The first index of the indices must be the index of the 
   *        middle_vector. The rest of the indices as well as the 
   *        rest_state must be given in clockwise order.
   * @param vertex_indices
   * @param middle_vertex_rest_state
   * @param rest_state
   * @param weight
   */
  BendingConstraint(const std::vector<std::size_t>& vertex_indices,
                    const Eigen::Vector3d& middle_vertex_rest_state,
                    const std::vector<Eigen::Vector3d>& rest_state,
                    double weight);
  /**
   * @brief ...
   * @param vertex_indices
   * @param middle_vertex_rest_state
   * @param rest_state
   * @param weight
   */
  BendingConstraint(const std::vector<std::size_t>& vertex_indices,
                    Eigen::Vector3d&& middle_vertex_rest_state,
                    std::vector<Eigen::Vector3d>&& rest_state,
                    double weight);
  /**
   * @brief ...
   * @param vertex_indices
   * @param middle_vertex_rest_state
   * @param rest_state
   * @param weight
   */
  BendingConstraint(std::vector<std::size_t>&& vertex_indices,
                    const Eigen::Vector3d& middle_vertex_rest_state,
                    const std::vector<Eigen::Vector3d>& rest_state,
                    double weight);
  /**
   * @brief ...
   * @param vertex_indices
   * @param middle_vertex_rest_state
   * @param rest_state
   * @param weight
   */
  BendingConstraint(std::vector<std::size_t>&& vertex_indices,
                    Eigen::Vector3d&& middle_vertex_rest_state,
                    std::vector<Eigen::Vector3d>&& rest_state,
                    double weight);

  /**
   * @brief Copy constructor
   * @param other
   */
  BendingConstraint(const BendingConstraint& other) = default;
  /**
   * @param Move constructor
   * @param other
   */
  BendingConstraint(BendingConstraint&& other) = default;

public:
  
  /**
   * @brief Request new lines in the sparse matrix containing the constraints
   *        (of size then ring1 x nVertices)
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

private:
  Eigen::Vector3d applyA(const Eigen::VectorXd& relevant_vector) const noexcept;

  /// @brief Norm of the Laplacian of the rest shape
  double m_rest_state_laplace_beltrami_norm;
  /// @brief Normalization coefficient for the Laplacian-Beltrami
  double m_laplace_beltrami_coefficient_sum;
  std::vector<double> m_beltrami_coefficients;

  /// @brief A in the PD potentiels
  Matrix m_A;
  /// @brief A^T A
  Matrix m_ATA;
};

#endif // BENDING_CONSTRAINT_HPP
