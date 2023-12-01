// vim: set ft=cpp :
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

#include "geometry/Mesh.hpp"
#include "util.hpp"

template<int Mode, int Option>
void
Mesh::applyTransformation(const Eigen::Transform<double, 3, Mode, Option>& transform) noexcept
{
    for (const auto& vertex_index : getVerticesRange())
    {
        getVector3dBlock(m_positions, vertex_index) =
          transform * getVector3dBlock(m_positions, vertex_index);
    }
}
