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
#ifndef PHYSICMESH_HPP
#define PHYSICMESH_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

#include <tiny_obj_loader.h>

#include "geometry/Mesh.hpp"
#include "physics/MaterialObject.hpp"
#include "physics/SpringConstraint.hpp"

/// @brief ...
struct PhysicParameters
{
  /// @brief Bending stiffness
  double bend;
  /// @brief Stretch stiffness
  double stretch;
  /// @brief Mass per surface unit
  double area_density;
};

/**
 * @brief Simulable physical mesh
 */
class PhysicMesh
  : public Mesh
  , public MaterialObject
{
public:
  /**
   * @brief Copy constructor
   * @param other
   */
  PhysicMesh(const PhysicMesh& other) = default;
  /**
   * @brief Move constructor
   * @param
   */
  PhysicMesh(PhysicMesh&& other) = default;

  /**
   * @brief Constructor
   * @param positions           Vertices positions
   * @param triangles           Triangles list
   * @param physic_parameters  
   * @param material_identifier Used to select the friction coefficient
   */
  PhysicMesh(const std::vector<Eigen::Vector3d>& positions,
             const std::vector<tinyobj::index_t>& triangles,
             const PhysicParameters& physic_parameters,
             std::size_t material_identifier);

  /**
   * @brief Getter
   * @return The constraints on the mesh
   */
  const std::vector<std::shared_ptr<AbstractConstraint>>&
  getConstraints() const;

  /**
   * @brief Getter
   * @return The speed
   */
  const Eigen::VectorXd& getGeneralizedSpeed() const;
  Eigen::Vector3d getVertexSpeed(size_t index) const;
  /**
   * @brief Setter
   * @param speed
   */
  void setGeneralizedSpeed(const Eigen::VectorXd& speed);
  /**
   * @brief Getter
   * @param vertex_index
   * @return Vertex mass
   */
  double getVertexMass(size_t vertex_index) const;

private:

  /**
   * @brief Helper - Range and store all the constraints
   */
  void initializeConstraints();
  /**
   * @brief Helper 
   * @param vertex_index
   * @return The vertex constraint
   */
  std::unique_ptr<AbstractConstraint> getVertexConstraint(
    std::size_t vertex_index) const;
  /**
   * @brief Helper
   * @param edge
   * @return The edge constraint
   */
  std::unique_ptr<AbstractConstraint> getEdgeConstraint(
    const Edge& edge) const;

  /// @brief Helper - Compute the vertices masses
  void initializeMasses();

  /**
   * @brief Compute an iterator on the faces adjacent to a vertex
   * @param vertex_index
   * @return The iterator
   */
  boost::iterator_range<
    boost::transform_iterator<std::function<double(const Triangle&)>,
                              TriangleIterator>>
    getTriangleAroundVertexMassRange(std::size_t vertex_index) const;

  /**
   * @brief Helper 
   * @return getTriangleMass
   */
  std::function<double(const Triangle&)> getTriangleMassFunctionObject()
    const;
  /**
   * @brief Compute a triangle mass
   * @param triangle
   */
  double getTriangleMass(const Triangle& triangle) const;

  /// @brief ...
  std::vector<std::shared_ptr<AbstractConstraint>> m_constraints;
  /// @brief ...
  PhysicParameters m_physic_parameters;
  /// @brief ...
  std::vector<double> m_masses;
  /// @brief ...
  Eigen::VectorXd m_speed;
};

#endif
