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
#ifndef IO_HPP
#define IO_HPP

/** @file
 * @brief Input/Output functions for files containing geometric objects.
 *
 * Currently, this file only contains I/O functions for OBJ meshes.
 */

#include "Mesh.hpp"

#include <Eigen/Core>
#include <string>
#include <vector>

#include <tiny_obj_loader.h>

/**@brief Load mesh from an OBJ file.
 *
 * This function opens an OBJ mesh file to load the described mesh.
 *
 * @param filename The path to the mesh file.
 * @return The loaded mesh.
 */
Mesh read_obj(const std::string& filename);

/**@brief Load generalized positions from an OBJ file.
 *
 * This function opens an OBJ mesh file to load the position of its vertices.
 * This is useful when multiple OBJ files describe the same mesh in different
 * positions.
 *
 * @param filename The path to the mesh file.
 * @return The generalized positions.
 */
Eigen::VectorXd read_obj_generalized_positions(const std::string& filename);

/**@brief Collect mesh data from an OBJ file.
 *
 * This function opens an OBJ mesh file to collect information such
 * as vertex position, vertex indices of a face, vertex normals and vertex
 * texture coordinates.
 *
 * @param filename The path to the mesh file.
 * @param positions The vertex positions.
 * @param indices The vertex indices of faces.
 * @param normals The vertex normals.
 * @param texcoords The vertex texture coordinates.
 * @return False if import failed, true otherwise.
 */
bool read_obj(const std::string& filename,
              std::vector<Eigen::Vector3d>& positions,
              std::vector<tinyobj::index_t>& indices,
              std::vector<Eigen::Vector3d>& normals,
              std::vector<Eigen::Vector2d>& texcoords);



#endif // IO_HPP
