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
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/iterator/filter_iterator.hpp>
#include <functional>
#include <vector>

#include <tiny_obj_loader.h>

#include "geometry/Mesh.hpp"
#include "util.hpp"

Mesh::Mesh(const std::vector<Eigen::Vector3d>& positions,
           const std::vector<tinyobj::index_t>& triangles)
  : m_vertices_indices(positions.size()), m_positions(positions.size() * 3)
{
    using VertexIndex = CGAL::Surface_mesh<Eigen::Vector3d>::Vertex_index;

    VertexIndex current_vertex_index;
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        current_vertex_index = m_mesh.add_vertex(positions[i]);
        // We want to be sure that the indices will stay inside the generalized
        // position ranged.
        assert(current_vertex_index < positions.size());
        m_vertices_indices[i] = current_vertex_index;
        getVector3dBlock(this->m_positions, current_vertex_index) = positions[i];
    }

    for (std::size_t i = 0; i < triangles.size() / 3; ++i)
    {
        // TODO: Auxiliary function to get triangle edges
        m_mesh.add_face(
          (SurfaceMesh::Vertex_index)m_vertices_indices[triangles[3 * i].vertex_index],
          (SurfaceMesh::Vertex_index)m_vertices_indices[triangles[3 * i + 1].vertex_index],
          (SurfaceMesh::Vertex_index)m_vertices_indices[triangles[3 * i + 2].vertex_index]);
    }
}

Mesh::Mesh(const SurfaceMesh& mesh) : m_mesh(mesh)
{
    m_vertices_indices.resize(m_mesh.number_of_vertices());
    std::iota(m_vertices_indices.begin(), m_vertices_indices.end(), 0);
    for (const auto& vertex_index : m_mesh.vertices())
    {
        getVector3dBlock(m_positions, vertex_index) = m_mesh.point(vertex_index);
    }
}

const Mesh::SurfaceMesh&
Mesh::getUnderlyingMesh() const
{
    return m_mesh;
}

void
Mesh::updateUnderlyingMesh()
{
    for (auto& vertex_index : m_mesh.vertices())
    {
        m_mesh.point(vertex_index) = getVector3dBlock(m_positions, vertex_index);
    }
}

std::size_t
Mesh::getNumberOfVertices() const
{
    return getVector3dSize(getGeneralizedPositions());
}

std::size_t
Mesh::getNumberOfTriangles() const noexcept
{
    return m_mesh.number_of_faces();
}

void Mesh::writeObj(std::ofstream &of)
{
  for (size_t vId = 0u; vId < getNumberOfVertices(); ++vId)
  {
    const Eigen::Vector3d v = getVertex(vId);
    of << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
  }
  for (size_t fId = 0u; fId < getNumberOfTriangles(); ++fId)
  {
    const Mesh::Triangle t = getTriangle(fId);
    of << "f " << t.vertex_indices[0] + 1
       << " " << t.vertex_indices[1] + 1
       << " " << t.vertex_indices[2] + 1 << std::endl;
  }
}


double
Mesh::getTriangleArea(const Triangle& triangle) const
{
    return std::sqrt(CGAL::squared_area<EigenKernel>(getVertex(triangle.vertex_indices[0]),
                                                     getVertex(triangle.vertex_indices[1]),
                                                     getVertex(triangle.vertex_indices[2])));
}

Eigen::Vector3d
Mesh::getTriangleNormal(const Triangle& triangle) const
{
    return (getVertex(triangle.vertex_indices[1]) - getVertex(triangle.vertex_indices[0]))
      .cross(getVertex(triangle.vertex_indices[2]) - getVertex(triangle.vertex_indices[0]))
      .normalized();
}

CGAL::Bbox_3
Mesh::getTriangleBoundingBox(const Triangle& triangle) const
{
    return EigenKernel::Triangle_3(getVertex(triangle.vertex_indices[0]),
                                   getVertex(triangle.vertex_indices[1]),
                                   getVertex(triangle.vertex_indices[2]))
      .bbox();
}

/**
 *  @brief Return the position of the triangle vertices in the column of a matrix.
 */
Eigen::Matrix3d
Mesh::getTriangle(const Triangle& triangle) const noexcept
{
    Eigen::Matrix3d result;
    result.col(0) = getVertex(triangle.vertex_indices[0]);
    result.col(1) = getVertex(triangle.vertex_indices[1]);
    result.col(2) = getVertex(triangle.vertex_indices[2]);
    return result;
}

Mesh::Triangle
Mesh::getTriangle(std::size_t triangle_index) const noexcept
{
    return getTriangleFromFaceIndex(SurfaceMesh::Face_index(triangle_index));
};

double
Mesh::getEdgeLength(const Mesh::Edge& edge) const
{
    return (getVertex(edge.vertex1_index) - getVertex(edge.vertex2_index)).norm();
}

Mesh::AllTriangleRange
Mesh::getTriangleRange() const noexcept
{
    return AllTriangleRange(getTriangleBegin(), getTriangleEnd());
}

Mesh::AllTriangleIterator
Mesh::getTriangleBegin() const noexcept
{
    return AllTriangleIterator(m_mesh.faces_begin(), getTriangleFromFaceIndexFunctionObject());
}

Mesh::AllTriangleIterator
Mesh::getTriangleEnd() const noexcept
{
    return AllTriangleIterator(m_mesh.faces_end(), getTriangleFromFaceIndexFunctionObject());
}

Mesh::TriangleRange
Mesh::getTriangleAroundVertexRange(std::size_t vertex_index) const
{
    return TriangleRange(getTriangleAroundVertexBegin(vertex_index),
                         getTriangleAroundVertexEnd(vertex_index));
}

Mesh::TriangleIterator
Mesh::getTriangleAroundVertexBegin(std::size_t vertex_index) const
{
    using FaceAroundTargetExceptSomeIterator =
      boost::filter_iterator<std::function<bool(const SurfaceMesh::Face_index&)>,
                             SurfaceMesh::Face_around_target_iterator>;

    auto triangle_around_vertex_range =
      m_mesh.faces_around_target(m_mesh.halfedge((SurfaceMesh::Vertex_index)vertex_index));
    return TriangleIterator(FaceAroundTargetExceptSomeIterator(
                              std::bind(&Mesh::isNotNullFace, this, std::placeholders::_1),
                              boost::begin(triangle_around_vertex_range),
                              boost::end(triangle_around_vertex_range)),
                            getTriangleFromFaceIndexFunctionObject());
}

Mesh::TriangleIterator
Mesh::getTriangleAroundVertexEnd(std::size_t vertex_index) const
{
    using FaceAroundTargetExceptSomeIterator =
      boost::filter_iterator<std::function<bool(const SurfaceMesh::Face_index&)>,
                             SurfaceMesh::Face_around_target_iterator>;

    auto triangle_around_vertex_range =
      m_mesh.faces_around_target(m_mesh.halfedge((SurfaceMesh::Vertex_index)vertex_index));
    return TriangleIterator(FaceAroundTargetExceptSomeIterator(
                              std::bind(&Mesh::isNotNullFace, this, std::placeholders::_1),
                              boost::end(triangle_around_vertex_range),
                              boost::end(triangle_around_vertex_range)),
                            getTriangleFromFaceIndexFunctionObject());
}

Mesh::EdgeRange
Mesh::getEdgeRange() const
{
    return EdgeRange(getEdgeBegin(), getEdgeEnd());
}

Mesh::EdgeIterator
Mesh::getEdgeBegin() const
{
    return EdgeIterator(m_mesh.edges_begin(), getEdgeFromEdgeIndexFunctionObject());
}

Mesh::EdgeIterator
Mesh::getEdgeEnd() const
{
    return EdgeIterator(m_mesh.edges_end(), getEdgeFromEdgeIndexFunctionObject());
}

Mesh::InteriorVerticesIterator
Mesh::getInteriorVerticesBegin() const
{
    return InteriorVerticesIterator(std::bind(&Mesh::isInteriorVertex, this, std::placeholders::_1),
                                    boost::begin(m_mesh.vertices()),
                                    boost::end(m_mesh.vertices()));
}

Mesh::InteriorVerticesIterator
Mesh::getInteriorVerticesEnd() const
{
    return InteriorVerticesIterator(std::bind(&Mesh::isInteriorVertex, this, std::placeholders::_1),
                                    boost::end(m_mesh.vertices()),
                                    boost::end(m_mesh.vertices()));
}

Mesh::VerticesAroundVertexRange
Mesh::getVerticesAroundVertexRange(std::size_t vertex_index) const
{
    return m_mesh.vertices_around_target(m_mesh.halfedge((SurfaceMesh::Vertex_index)vertex_index));
}

Mesh::VerticesRange
Mesh::getVerticesRange() const
{
    return m_mesh.vertices();
}

std::vector<Eigen::Vector3d>
Mesh::getVertices() const noexcept
{
    std::vector<Eigen::Vector3d> result(getNumberOfVertices());
    for (std::size_t vertex_index : getVerticesRange())
    {
        result[vertex_index] = getVertex(vertex_index);
    }
    return result;
}

Mesh::Vertex
Mesh::getVertex(std::size_t index) const
{
    return getVector3dBlock(getGeneralizedPositions(), index);
}

bool
Mesh::isVertexOnBorder(std::size_t vertex_index) const
{
    return m_mesh.is_border((SurfaceMesh::Vertex_index)vertex_index);
}

bool
Mesh::isInteriorVertex(std::size_t vertex_index) const
{
    return !isVertexOnBorder(vertex_index);
}

const Eigen::VectorXd&
Mesh::getGeneralizedPositions() const
{
    return m_positions;
}

Eigen::Vector3d
Mesh::getVertexPosition(size_t index) const
{
  return getVector3dBlock(getGeneralizedPositions(), index);
}

void
Mesh::setGeneralizedPositions(const Eigen::VectorXd& q)
{
    m_positions = q;
}

void
Mesh::setGeneralizedPositions(Eigen::VectorXd&& q)
{
    m_positions = std::move(q);
}

void
Mesh::setPositions(const std::vector<Eigen::Vector3d>& positions)
{
    setPositions(std::vector<Eigen::Vector3d>(positions));
}

void
Mesh::setPositions(std::vector<Eigen::Vector3d>&& positions)
{
    assert(positions.size() == getNumberOfVertices());
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        getVector3dBlock(m_positions, m_vertices_indices[i]) = positions[i];
    }
}

std::function<Mesh::Edge(const Mesh::SurfaceMesh::Edge_index&)>
Mesh::getEdgeFromEdgeIndexFunctionObject() const
{
    return std::bind(&Mesh::getEdgeFromEdgeIndex, this, std::placeholders::_1);
}

Mesh::Edge
Mesh::getEdgeFromEdgeIndex(const SurfaceMesh::Edge_index& edge_index) const
{
    return Edge{ m_mesh.vertex(edge_index, 0), m_mesh.vertex(edge_index, 1) };
}

std::function<Mesh::Triangle(const Mesh::SurfaceMesh::Face_index&)>
Mesh::getTriangleFromFaceIndexFunctionObject() const
{
    return std::bind(&Mesh::getTriangleFromFaceIndex, this, std::placeholders::_1);
}

Mesh::Triangle
Mesh::getTriangleFromFaceIndex(const SurfaceMesh::Face_index& face_index) const
{
    auto vertices_around_face_range = m_mesh.vertices_around_face(m_mesh.halfedge(face_index));

    Triangle triangle;
    std::copy(boost::begin(vertices_around_face_range),
              boost::end(vertices_around_face_range),
              triangle.vertex_indices.begin());

    return triangle;
}

bool
Mesh::isNotNullFace(const SurfaceMesh::Face_index& face_index) const
{
    return face_index != m_mesh.null_face();
}
