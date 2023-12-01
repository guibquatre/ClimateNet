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
#ifndef SPRING_CONSTRAINT_HPP
#define SPRING_CONSTRAINT_HPP

#include "physics/AbstractConstraint.hpp"

/// @brief Class implementing a spring between two vertex
class SpringConstraint : public AbstractConstraint
{
private:

  ///@brief Number of vertices involved = 2
  static const std::size_t m_nb_position_involved;


public:

  /**
   * @brief Constructor
   * @param vertex1_index  Vertex index of one end of the spring
   * @param vertex2_index  Vertex index of the other end of the spring
   * @param weight         Weight of the constraint
   * @param rest_length    Rest length of the spring
   */
  SpringConstraint(std::size_t vertex1_index,
                   std::size_t vertex2_index,
                   double weight,
                   double rest_length);

  /**
   * @brief Copy constructor
   * @param other
   */
  SpringConstraint(const SpringConstraint& other) = default;

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

private:

  /// @brief Spring rest length
  double m_rest_length;
};

#endif //SPRING_CONSTRAINT_HPP
