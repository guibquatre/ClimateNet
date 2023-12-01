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
#ifndef OBSTACLE_HPP
#define OBSTACLE_HPP

#include <vector>

#include <Eigen/Core>

#include "physics/MaterialObject.hpp"

/// @brief Struct containing the data needed to handle a collision
struct CollisionInfo 
{
  /// @brief (Generalized) index of the vertex in contact.
  size_t vertex_index;
  /// @brief Position of the contact point.
  Eigen::Vector3d contact_point;
  /// @brief Normal at the contact point.
  Eigen::Vector3d normal;
  /// Speed of the contact point.
  Eigen::Vector3d speed;
  /// @brief Friction coefficient of the two material in collision.
  double friction_coefficient;
};

/**
 * Stores the date required to handle a self-collision. Currently,
 * self-collision is only node-node.
 */
struct SelfCollisionInfo : public CollisionInfo
{
  /**
   * Index of the vertices that make up the triangle with which the vertex
   * collided.
   */
  std::array<size_t, 3> face_indices;
  /**
   * The barycentric coordinate of the point in the face with which the vertex
   * has collided. Currently, since we only handle node-node self-collision,
   * this vector has a coordiante set to 1 and the other to 0.
   */
  Eigen::Vector3d barycentric_coordinates;
};

/// @brief Abstract class for an obstacle
class Obstacle : public MaterialObject
{
public:

  using MaterialObject::MaterialObject;
  virtual ~Obstacle() = default;

  virtual void update(double step) = 0;

  /**
   *    @bried Append the contact point, normals at contact point and speed at contact point to the
   *    given vectors for every given vertex in contact.
   */
  virtual void checkCollision(
          const std::vector<Eigen::Vector3d>& vertices,
          std::vector<size_t>& vertices_indices,
          std::vector<Eigen::Vector3d>& contact_points,
          std::vector<Eigen::Vector3d>& normals,
          std::vector<Eigen::Vector3d>& speeds) const noexcept;

  /**
   * @brief Check a collision
   * @param[in]  pos            Position to check
   * @param[out] contact_point  Contact point (if valid)
   * @param[out] normal         Normal at the contact point (if valid)
   * @param[out] speed          Speed at the contact point (if valid)
   * @return true if there is a contact (thus contact_point and normal are valid)
   */
  virtual bool checkCollision(const Eigen::Vector3d &pos,
          Eigen::Vector3d& contact_point,
          Eigen::Vector3d& normal,
          Eigen::Vector3d& speed) const noexcept = 0;
};

#endif // OBSTACLE
