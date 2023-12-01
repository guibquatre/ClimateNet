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
#include "physics/StaticMesh.hpp"
#include "geometry/geometry_util.hpp"
#include <Eigen/Geometry>

StaticMesh::StaticMesh(const std::shared_ptr<Mesh> meshPtr, double tol, size_t material_identifier)
  : StaticObstacle(material_identifier), m_meshPtr(meshPtr), m_tol2(tol * tol)
{
}

/**
 *    @bried Append the contact point, normals at contact point and speed at contact point to
 * the given vectors for every given vertex in contact.
 */
void
StaticMesh::checkCollision(const std::vector<Eigen::Vector3d>& vertices,
                           std::vector<size_t>& vertices_indices,
                           std::vector<Eigen::Vector3d>& contact_points,
                           std::vector<Eigen::Vector3d>& normals) const noexcept
{
    checkMeshCollision(
            vertices,
            *m_meshPtr,
            std::sqrt(m_tol2),
            vertices_indices,
            contact_points,
            normals,
            nullptr,
            nullptr);
}

bool
StaticMesh::checkCollision(const Eigen::Vector3d& pos,
                           Eigen::Vector3d& contact_point,
                           Eigen::Vector3d& normal) const noexcept
{
    return checkMeshCollision(pos, *m_meshPtr, m_tol2, contact_point, normal);
}

