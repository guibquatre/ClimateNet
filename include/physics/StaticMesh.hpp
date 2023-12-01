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
#ifndef STATIC_MESH_HPP
#define STATIC_MESH_HPP
#include "geometry/Mesh.hpp"
#include "physics/StaticObstacle.hpp"
#include <memory>

class StaticMesh : public StaticObstacle
{
  public:
    /**
     * @brief Constructor
     * @param mesh  Mesh to be used as an obstacle
     * @param tol   Tolerance for the contact
     * @param material_identifier
     */
    StaticMesh(const std::shared_ptr<Mesh> meshPtr, double tol, size_t material_identifier);

    /**
     *    @bried Append the contact point, normals at contact point and speed at contact point to
     * the given vectors for every given vertex in contact.
     */
    virtual void checkCollision(const std::vector<Eigen::Vector3d>& vertices,
                                std::vector<size_t>& vertices_indices,
                                std::vector<Eigen::Vector3d>& contact_points,
                                std::vector<Eigen::Vector3d>& normals) const
      noexcept override final;

    /**
     * @brief Check a collision
     * @param[in]  pos            Position to check
     * @param[out] contact_point  Contact point (if valid)
     * @param[out] normal         Normal at the contact point (if valid)
     * @return true if there is a contact (thus contact_point and normal are valid)
     */
    bool checkCollision(const Eigen::Vector3d& pos,
                        Eigen::Vector3d& contact_point,
                        Eigen::Vector3d& normal) const noexcept override final;

  private:
    const std::shared_ptr<Mesh> m_meshPtr;
    double m_tol2;
};

#endif // STATIC_MESH_HPP

