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
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include <tiny_obj_loader.h>
#undef TINYOBJLOADER_IMPLEMENTATION


#include "geometry/io.hpp"
#include "util.hpp"

#define DIMENSION 3
#define NB_VERTICES_IN_TRIANGLE 3

Mesh read_obj(const std::string& filename)
{
      std::vector<tinyobj::index_t> triangles;
      std::vector<Eigen::Vector3d> vertices;
      std::vector<Eigen::Vector3d> normals;
      std::vector<Eigen::Vector2d> texcoords;

    bool success = read_obj(filename, vertices, triangles, normals, texcoords);
    if (!success)
    {
        throw std::invalid_argument("Can't read obj file " + filename);
    }

    return Mesh(vertices, triangles);
}

Eigen::VectorXd read_obj_generalized_positions(const std::string& filename)
{
      std::vector<tinyobj::index_t> triangles;
      std::vector<Eigen::Vector3d> vertices;
      std::vector<Eigen::Vector3d> normals;
      std::vector<Eigen::Vector2d> texcoords;

    bool success = read_obj(filename, vertices, triangles, normals, texcoords);
    if (!success)
    {
        throw std::invalid_argument("Can't read obj file " + filename);
    }

    return getVectorConcatenation(vertices.begin(), vertices.end());
}

bool
read_obj(const std::string& filename,
         std::vector<Eigen::Vector3d>& positions,
         std::vector<tinyobj::index_t>& triangles,
         std::vector<Eigen::Vector3d>& normals,
         std::vector<Eigen::Vector2d>& texcoords)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    bool ret =
      tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        return ret;
    }

    positions.clear();
    triangles.clear();
    normals.clear();
    texcoords.clear();

    for (size_t i = 0; i < shapes.size(); i++) {
        assert(shapes[i].mesh.indices.size() % NB_VERTICES_IN_TRIANGLE == 0);
        for (size_t f = 0; f < shapes[i].mesh.indices.size(); f++) {
            triangles.push_back(shapes[i].mesh.indices[f]);
        }
        assert(attrib.vertices.size() % DIMENSION == 0);
        for (size_t v = 0; v < attrib.vertices.size() / DIMENSION; v++) {
            positions.push_back(
              Eigen::Vector3d(attrib.vertices[DIMENSION * v + 0],
                              attrib.vertices[DIMENSION * v + 1],
                              attrib.vertices[DIMENSION * v + 2]));
        }

        assert(attrib.normals.size() % DIMENSION == 0);
        for (size_t n = 0; n < attrib.normals.size() / 3; n++) {
            normals.push_back(Eigen::Vector3d(attrib.normals[3 * n + 0],
                                              attrib.normals[3 * n + 1],
                                              attrib.normals[3 * n + 2]));
        }

        assert(attrib.texcoords.size() % (DIMENSION - 1) == 0);
        for (size_t t = 0; t < attrib.texcoords.size() / 2; t++) {
            texcoords.push_back(Eigen::Vector2d(attrib.texcoords[2 * t + 0],
                                                attrib.texcoords[2 * t + 1]));
        }
    }

    return ret;
}

