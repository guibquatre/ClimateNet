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
#include "geometry/Sphere.hpp"

Sphere::Sphere(double radius, const Eigen::Vector3d& center)
  : Sphere(radius, Eigen::Vector3d(center))
{
}
Sphere::Sphere(double radius, Eigen::Vector3d&& center)
  : m_radius(radius), m_center(std::move(center))
{
}

Mesh
Sphere::getMesh(double number_sector, double number_stack) const noexcept
{
    std::vector<Eigen::Vector3d> vertices;
    std::vector<tinyobj::index_t> triangles;

    Eigen::Vector3d current_vertex;
    double xy;

    float sectorStep = 2 * M_PI / number_sector;
    float stackStep = M_PI / number_stack;
    float sectorAngle, stackAngle;

    for(int i = 0; i <= number_stack; ++i)
    {
        stackAngle = M_PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
        xy = m_radius * std::cos(stackAngle);             // r * cos(u)
        current_vertex.z() = m_radius * std::sin(stackAngle);              // r * sin(u)

        // add (number_sector+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for(int j = 0; j <= number_sector; ++j)
        {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position (x, y, z)
            current_vertex.x() = xy * std::cos(sectorAngle);             // r * cos(u) * cos(v)
            current_vertex.y() = xy * std::sin(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(m_center + current_vertex);
        }
    }

    int k1, k2;
    for(int i = 0; i < number_stack; ++i)
    {
        k1 = i * (number_sector + 1);     // beginning of current stack
        k2 = k1 + number_sector + 1;      // beginning of next stack

        for(int j = 0; j < number_sector; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1+1
            if(i != 0)
            {
                triangles.push_back(tinyobj::index_t{k1, 0, 0});
                triangles.push_back(tinyobj::index_t{k2, 0, 0});
                triangles.push_back(tinyobj::index_t{k1 + 1, 0, 0});
            }

            // k1+1 => k2 => k2+1
            if(i != (number_stack-1))
            {
                triangles.push_back(tinyobj::index_t{k1 + 1, 0, 0});
                triangles.push_back(tinyobj::index_t{k2, 0, 0});
                triangles.push_back(tinyobj::index_t{k2 + 1, 0, 0});
            }
        }
    }

    return Mesh(vertices, triangles);
}

bool
Sphere::isInside(const Eigen::Vector3d& point) const noexcept
{
    Eigen::Vector3d r = point - m_center;
    return r.norm() < m_radius + 1e-2;
}

/**
 *  @brief Stores the projection of \a point on the sphere surface in \a projection and the
 *  normal at the the projection in \a normal.
 */
void
Sphere::project(const Eigen::Vector3d& point,
                Eigen::Vector3d& projection,
                Eigen::Vector3d& normal) const noexcept
{
    Eigen::Vector3d r = point - m_center;
    r.normalize();
    projection = m_center + m_radius * r;
    normal = r;
}

const Eigen::Vector3d&
Sphere::getCenter() const noexcept
{
    return m_center;
}

double
Sphere::getRadius() const noexcept
{
    return m_radius;
}

