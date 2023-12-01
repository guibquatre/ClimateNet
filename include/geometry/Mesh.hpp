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
#ifndef MESH_HPP
#define MESH_HPP

#include <CGAL/Surface_mesh.h>
#include <boost/iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/range/sub_range.hpp>
#include <Eigen/Core>
#include <functional>
#include <tiny_obj_loader.h>
#include <vector>
#include <fstream>

#include "eigen_kernel.hpp"

/** @class Mesh
 * A surface mesh made of triangles.
 */

class Mesh
{
public:
    using SurfaceMesh = CGAL::Surface_mesh<Eigen::Vector3d>;
  
    /**
     * Type of an iterator over the interior vertices of the mesh.
     * @see getInteriorVerticesEnd
     * @see getInteriorVerticesBegin
     */
    using InteriorVerticesIterator = boost::filter_iterator<
        std::function<bool(std::size_t)>,
        SurfaceMesh::Vertex_iterator>;
    /**
     * Type of a range over the indices of vertices adjacent to a vertex.
     * @see getVerticesAroundVertexRange
     */
    using VerticesAroundVertexRange =
      CGAL::Iterator_range<CGAL::Vertex_around_target_iterator<SurfaceMesh>>;
    /**
     * Type of a range over the indices of the mesh's vertices.
     */
    using VerticesRange = SurfaceMesh::Vertex_range;
    using Vertex = Eigen::Vector3d; /**< Type of the vertices position \see getVertex */

    /**
     * An edge of the mesh.
     */
    struct Edge
    {
        /** Stores the indices of the vertices the edge is made of. */
        std::size_t vertex1_index, vertex2_index;
    };

    /**
     * The type of an iterator over every edge within the mesh.
     * @see getEdgeBegin
     * @see getEdgeEnd
     */
    using EdgeIterator = boost::transform_iterator<
      std::function<Edge(const SurfaceMesh::Edge_index&)>,
      SurfaceMesh::Edge_iterator>;
    /**
     * The type of a range over every edge within the mesh. 
     * @see getEdgeRange
     */
    using EdgeRange = boost::iterator_range<EdgeIterator>;

    /**
     * A triangle of the mesh.
     */
    struct Triangle
    {
        /** Stores the indices of the vertices the triangle is made of */
        std::array<std::size_t, 3> vertex_indices;
    };
    /**
     * The type of an iterator over every triangle within the mesh.
     * @see getTriangleBegin
     * @see getTriangleEnd
     */
    using AllTriangleIterator = boost::transform_iterator<
      std::function<Triangle(const SurfaceMesh::Face_index&)>,
        SurfaceMesh::Face_iterator>;
    /**
     * The type of a range over every triangle within the mesh.
     * @see getTriangleRange
     */
    using AllTriangleRange = boost::iterator_range<AllTriangleIterator>;
    using TriangleIterator = boost::transform_iterator<
      std::function<Triangle(const SurfaceMesh::Face_index&)>,
      boost::filter_iterator<
        std::function<bool(const SurfaceMesh::Face_index&)>,
        SurfaceMesh::Face_around_target_iterator>>;
    using TriangleRange = boost::iterator_range<TriangleIterator>;

    Mesh(const Mesh& other) = default;
    Mesh(Mesh&& other) = default;

    /**
     * Contruct a mesh from a set of ordered vertices and a set of triangles.
     *
     * @param positions Vector of orderd vertices. The order is used to know
     *                  which vertices are in a triangle.
     *
     * @param triangles Vector of triangle description following
     *                  tiny_obj_reader usage.
     *
     * @see tiny_obj_reader.h
     */
    Mesh(const std::vector<Eigen::Vector3d>& positions,
         const std::vector<tinyobj::index_t>& triangles);
    Mesh(const SurfaceMesh& surface_mesh);

    /**
     * Apply a tranformation on each vertices of the mesh.
     * @param transform The transformation to apply.
     */
    template<int Mode, int Option>
    void applyTransformation(const Eigen::Transform<double, 3, Mode, Option>& transform) noexcept;

    /**
     * Returns the CGAL Surface Mesh associated to the mesh. This method is
     * mostly used for outputing the data in a file. The returned mesh might
     * not be storing the up to date position of the vertices, you might want to
     * call the method updateUnderlyingMesh before calling this method.
     */
    const SurfaceMesh& getUnderlyingMesh() const;
    /**
     * Updates the position of the vertices stored in the CGAL Mesh. This method
     * is mostly used for outputing data in a file in conjunction to
     * getUnderlyingMesh.
     */
    void updateUnderlyingMesh();

    /**
     * Returns the number of vertices in the mesh.
     */
    std::size_t getNumberOfVertices() const;
    /**
     * Returns the number of triangles in the mesh.
     */
    std::size_t getNumberOfTriangles() const noexcept;

    /**
     * Returns the area of the given triangle.
     */
    double getTriangleArea(const Triangle& triangle) const;
    /**
     * Returns the outward normal of the given triangle. The normal is obtained
     * through the normalized cross product of the edge (v0, v1) and the edge
     * (v0, v2) of the triangle (in this order). 
     */
    Eigen::Vector3d getTriangleNormal(const Triangle& triangle) const;
    /**
     * Returns the axis aligned bounding box of the given triangle.
     */
    CGAL::Bbox_3 getTriangleBoundingBox(const Triangle& triangle) const;
    /**
     *  @brief Return the position of the triangle vertices in the column of a
     *  matrix.  Denoting the result by `result`, `result.col(i)` is the
     *  position of the `i`th vertex of the triangle.
     */
    Eigen::Matrix3d getTriangle(const Triangle& triangle) const noexcept;
    /**
     * Returns the triangle which has the given index. 
     */
    Mesh::Triangle getTriangle(std::size_t triangle_index) const noexcept;

    /**
     * Returns a range over every triangle of the mesh. The returned range
     * iterates over object of type Triangle.
     * @getTriangleBegin
     * @getTriangleEnd
     */
    AllTriangleRange getTriangleRange() const noexcept;
    /**
     * Returns the starting iterator of a range over every triangles in the
     * mesh. This iterator iterates on object of type Triangle.
     * @see getTriangleRange
     * @see getTriangleEnd
     */
    AllTriangleIterator getTriangleBegin() const noexcept;
    /**
     * Returns the after end iterator of a range over every triangles in the
     * mesh.
     * @see getTriangleRange
     * @see getTriangleEnd
     */
    AllTriangleIterator getTriangleEnd() const noexcept;

    /**
     * Returns a range over every triangle incident to the vertex that has the
     * given index within the mesh. The returned range iterates over object of
     * type Triangle.
     * @see getTriangleAroundVertexBegin
     * @see getTriangleAroundVertexEnd
     */
    TriangleRange getTriangleAroundVertexRange(std::size_t vertex_index) const;
    /**
     * Returns the starting iterator of a range over every triangle incident to
     * the vertex that has the given index within the mesh. This iterator
     * iterates on object of type Triangle.
     * @see getTriangleAroundVertexEnd
     * @see getTriangleAroundVertexRange
     */
    TriangleIterator getTriangleAroundVertexBegin(
      std::size_t vertex_index) const;
    /**
     * Returns the after end iterator of a range over every triangle incident to
     * the vertex that has the given index within the mesh.
     * @see getTriangleAroundVertexBegin
     * @see getTriangleAroundVertexRange
     */
    TriangleIterator getTriangleAroundVertexEnd(std::size_t vertex_index) const;

    /**
     * Return the length of the given edge.
     */
    double getEdgeLength(const Edge& edge) const;

    /**
     * Returns a range over every edge of the mesh. The returned range
     * iterates over object of type Edge.
     * @getEdgeBegin
     * @getEdgeEnd
     */
    EdgeRange getEdgeRange() const;
    /**
     * Returns the starting iterator of a range over every edge in the
     * mesh. This iterator iterates on object of type Edge.
     * @see getEdgeRange
     * @see getEdgeEnd
     */
    EdgeIterator getEdgeBegin() const;
    /**
     * Returns the after end iterator of a range over every edge in the
     * mesh.
     * @see getEdgeRange
     * @see getEdgeBegin
     */
    EdgeIterator getEdgeEnd() const;

    /**
     * Returns the starting iterator of a range over the indices of the vertices
     * that are not on the a border of the mesh. A vertex is on a border if one
     * of its incident edge doesn't have two incident triangle.
     * @see getInteriorVerticesEnd
     * @see isVertexOnBorder
     * @see isInteriorVertex
     */
    InteriorVerticesIterator getInteriorVerticesBegin() const;
    /**
     * Returns the after end iterator of a range over the indices of the vertices
     * that are not on the a border of the mesh.
     * @see getInteriorVerticesBegin
     * @see isVertexOnBorder
     * @see isInteriorVertex
     */
    InteriorVerticesIterator getInteriorVerticesEnd() const;
    /**
     * Returns a range over the indices of every vertex adjacent to the vertex
     * whose index within the mesh is the given index.
     */
    VerticesAroundVertexRange getVerticesAroundVertexRange(
      std::size_t vertex_index) const;
    /**
     * Returns a range over the indices of every vertex of the mesh.
     */
    VerticesRange getVerticesRange() const;
    /**
     * Returns the position of every vertex within the mesh.
     */
    std::vector<Eigen::Vector3d> getVertices() const noexcept;

    /**
     * Return the position of the vertex whose index within the mesh is the
     * given index.
     */
    Vertex getVertex(std::size_t index) const;
    /**
     * Returns true if and only if the vertex with the given index within the
     * mesh is on the border of the mesh. A vertex is on the border if one of
     * its incident edge has only one incident triangle. 
     * @see isInteriorVertex
     * @see getInteriorVerticesEnd
     * @see getInteriorVerticesBegin
     */
    bool isVertexOnBorder(std::size_t vertex_index) const;
    /**
     * Returns true if and only if the vertex with the given index within the
     * mesh is not on the border of the mesh. A vertex is on the border if one
     * of its incident edge has only one incident triangle.
     * @see isVertexOnBorder
     * @see getInteriorVerticesEnd
     * @see getInteriorVerticesBegin
     */
    bool isInteriorVertex(std::size_t vertex_index) const;

    /**
     * Returns the position.
     *
     * The position for a mesh is simply the concatenation of
     * the position of each vertex into one row vector. The order is
     * preserved.
     */
    const Eigen::VectorXd& getGeneralizedPositions() const;
    Eigen::Vector3d getVertexPosition(size_t index) const;

    // TODO: Change the name to setPositions
    /**
     * Change the position.
     *
     * It copies the given vector.
     * \see getGeneralizedPosition
     */
    void setGeneralizedPositions(const Eigen::VectorXd& q);

    /**
     * Change the position.
     *
     * It moves the given vector.
     * \see getGeneralizedPosition
     */
    void setGeneralizedPositions(Eigen::VectorXd&& q);

    /**
     * Sets the position of the mesh.
     * @param positions The position of each vertices.
     */
    void setPositions(const std::vector<Eigen::Vector3d>& positions);
    /**
     * Sets the position of the mesh.
     * @param positions The position of each vertices.
     */
    void setPositions(std::vector<Eigen::Vector3d>&& positions);

    Mesh& operator=(Mesh&& other) = default;
    Mesh& operator=(const Mesh& other) = default;

  void writeObj(std::ofstream &of);

private:
    /**
     * Returns a function which takes the index of an edge and return the
     * associated Edge object.
     */
    std::function<Edge(const SurfaceMesh::Edge_index&)>
    getEdgeFromEdgeIndexFunctionObject() const;
    /**
     * Returns the Edge object associated to the given edge index.
     */
    Edge getEdgeFromEdgeIndex(const SurfaceMesh::Edge_index& edge_index) const;

    /**
     * Returns a function which takes the index of a triangle and returns the
     * associated Triangle object.
     */
    std::function<Triangle(const SurfaceMesh::Face_index&)>
    getTriangleFromFaceIndexFunctionObject() const;
    /**
     * Returns the Triangle object associated to the given triangle index.
     */
    Triangle getTriangleFromFaceIndex(
      const SurfaceMesh::Face_index& face_index) const;
    /**
     * Returns true if the index is a null face. A null face is used in CGAL to
     * represent the absence of a triangle. For example, an edge on the border
     * of the mesh is an edge that has one null face within its incident faces.
     */
    bool isNotNullFace(const SurfaceMesh::Face_index& face_index) const;

    /**
     * A CGAL mesh that stores the topology of the mesh. The position of the
     * vertices are not stored in this data member as it would be too costly.
     * To update the position within this data member, use updateUnderlyingMesh.
     */
    SurfaceMesh m_mesh;
    /**
     * A mapping from the vertices index passed to this class methods to the
     * vertices indices within the CGAL surface mesh.  Since the CGAL surface
     * mesh does not necessarily keep the vertices in the order in which they
     * were added to the mesh, we keep a mapping.
     */
    std::vector<std::size_t> m_vertices_indices;
    /**
     * The concatenated positions of the vertices.
     */
    Eigen::VectorXd m_positions;
};

// Include implementation of templated methods.
#include "geometry/Mesh.tpp"

#endif
