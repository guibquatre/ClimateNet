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
#include "physics/Obstacle.hpp"



void Obstacle::checkCollision(
          const std::vector<Eigen::Vector3d>& vertices,
          std::vector<size_t>& vertices_indices,
          std::vector<Eigen::Vector3d>& contact_points,
          std::vector<Eigen::Vector3d>& normals,
          std::vector<Eigen::Vector3d>& speeds) const noexcept
{
    Eigen::Vector3d contact_point;
    Eigen::Vector3d normal;
    Eigen::Vector3d speed;
    for (std::size_t vertex_index = 0; vertex_index < vertices.size(); ++vertex_index)
    {

        if (this->checkCollision(vertices[vertex_index], contact_point, normal, speed))
        {
            vertices_indices.push_back(vertex_index);
            contact_points.push_back(contact_point);
            normals.push_back(normal);
            speeds.push_back(speed);
        }
    }
}
