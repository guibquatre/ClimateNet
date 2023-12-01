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
#ifndef GEOMETRY_UTIL_HPP
#define GEOMETRY_UTIL_HPP

#include "geometry/Mesh.hpp"
#include <Eigen/Core>
#include <vector>
#include "geometry/Mesh.hpp"
#include "physics/Obstacle.hpp"

/**
 * @brief Check the presence of collisions between the given vertices and the
 *        given mesh. For each vertices, only the closest contact point is kept.
 * 
 * @param tolerance The tolerance of the collision detection. A point is
 *                  considered in contact with the mesh if its distance to the
 *                  mesh is less than the tolerance.
 * @param[in] vertices_indices Vector in which the colliding vertices index are
 *                             appended.
 * @param[in] contact_points Vector in which the contact points of collisions
 *                           are appended
 * @param[in] normals Vector in which the normals of the mesh at the contact
 *                    points are appended
 * @param[i] barycentric_coordinates_ptr If the pointer is not the null pointer
 *                                       the barycentric coordinate of the
 *                                       contact points in the triangle they
 *                                       belong to are appended.
 * @param[in] faces_indices_ptr If the pointer is not the null pointer the
 *                              indices of the vertices that make up the
 *                              triangles in which the contact points belong are
 *                              appended.
 */

void
checkMeshCollision(const std::vector<Eigen::Vector3d>& vertices,
                   const Mesh& mesh,
                   double tolerance,
                   std::vector<std::size_t>& vertices_indices,
                   std::vector<Eigen::Vector3d>& contact_points,
                   std::vector<Eigen::Vector3d>& normals,
                   std::vector<Eigen::Vector3d>* barycentric_coordinates_ptr,
                   std::vector<Mesh::Triangle>* faces_indices_ptr);

/**
 * Returns self-collision information for the given mesh. Since only node-node
 * self-collision is supported, each vertices of the mesh that is at less
 * than the given tolerance to a face is paired with one vertex of this face.
 * However, we make sure that a vertex is not linked with more than one vertex
 * per triangle. We introduced this implementation after seeing some
 * instabilities. However, these instabilities are probably all resolved by the
 * ordering of contacts 
 */
std::vector<SelfCollisionInfo>
checkMeshSelfCollisions(const Mesh &mesh,
                        double tolerance) noexcept;

std::size_t isTriangleVertexIndexIn(const Mesh::Triangle& triangle, const std::vector<size_t>& vertices_indices);

/**
 * @brief Find all collisions of the given vertices with the given mesh. The information on the
 * collisions of the vertex with index i are at the index i of the returned vector. The collision
 * information is stored in form of pairs of colliding triangle and barycentric coordinate of
 * contact point.
 */
std::vector<std::vector<std::pair<Mesh::Triangle, Eigen::Vector3d>>>
checkMeshAllCollision(const std::vector<Eigen::Vector3d>& vertices,
                   const Mesh& mesh,
                   double tolerance);

/**
 * @brief The triangle at index i are the ones with whom the vertex with index i potentially
 * collide. This uses CGAL efficient bounding box intersection finder box_intersection_d.
 */
std::vector<std::vector<Mesh::Triangle>>
getMeshPotentialTriangleCollision(const std::vector<Eigen::Vector3d>& vertices,
                                  const Mesh& mesh,
                                  double tolerance);

CGAL::Bbox_3
getToleranceBoundingBox(const Eigen::Vector3d& vertex, double tolerance);

/**
 * @brief From Christer Ericson -- Real-Time Collision Detection (p141)
 *        (From an implementation by Gilles Daviet)
 *
 * @param p     Vertex to project
 * @param triangle Matrix whose column are the positions of the triangle vertices
 * @param tol2  Tolerance for the collision
 * @param closest_point
 * @paral barycentric_coordinates_ptr
 */
void
closestPointToTriangle(const Eigen::Vector3d& p,
                       const Eigen::Matrix3d& triangle,
                       double tol2,
                       Eigen::Vector3d& closest_point,
                       Eigen::Vector3d* barycentric_coordinates_ptr = nullptr) noexcept;

bool
checkMeshCollision(const Eigen::Vector3d& pos,
                   const Mesh& mesh,
                   double tol2,
                   Eigen::Vector3d& contact_point,
                   Eigen::Vector3d& normal,
                   Eigen::Vector3d* barycentric_coordinates_ptr = nullptr,
                   // Activate the self-collision exception mode
                   const size_t *vertex_index_ptr = nullptr,
                   std::array<size_t, 3> *face_indices_ptr = nullptr)
  noexcept;

std::vector<SelfCollisionInfo>
checkMeshSelfCollisions(const Eigen::Vector3d &pos,
                        const Mesh &mesh,
                        double tol2,
                        size_t vId) noexcept;

Eigen::Matrix3Xd
getLaplaceBeltramiDicretisation(const Eigen::Vector3d& middle_vertex,
                                const std::vector<Eigen::Vector3d>& ring_vertices);

Eigen::MatrixXd
getReducedLaplaceBeltramiDicretisation(const Eigen::Vector3d& middle_vertex,
                                       const std::vector<Eigen::Vector3d>& ring_vertices);

std::vector<double>
getLaplaceBeltramiDiscretisationMeanValueCoefficients(
  const Eigen::Vector3d& middle_vertex,
  const std::vector<Eigen::Vector3d>& ring_vertices);

std::vector<double>
getLaplaceBeltramiDiscretisationCotangentCoefficients(
  const Eigen::Vector3d& middle_vertex,
  const std::vector<Eigen::Vector3d>& ring_vertices);

/**
 * Return a change of basis matrix into an orthonormal direct basis which have
 * the normalization of vector as one of its vectors. The matrix is a change
 * of basis matrix from local to world i-e if v is a local vector and P is
 * the returned matrix then Pv is v in world coordinate. The base could be
 * any of the infinitely many valid ones.
 */
Eigen::Matrix3d
getChangeOfBasisIntoOrthonormalDirectBasis(const Eigen::Vector3d& vector);

double
getMeanValue(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3);

double
getSine(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3);
double
getCosine(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3);

double
getCotangent(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3);

bool
areCollinear(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3);

#endif
