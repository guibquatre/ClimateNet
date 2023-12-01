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
#include "geometry/geometry_util.hpp"

#include "eigen_kernel.hpp"
#include <Eigen/Geometry>
#include <vector>

#include <CGAL/box_intersection_d.h>

#include "util.hpp"

#define PRECISION 1e-20

void
checkMeshCollision(const std::vector<Eigen::Vector3d>& vertices,
                   const Mesh& mesh,
                   double tolerance,
                   std::vector<std::size_t>& colliding_vertices_indices,
                   std::vector<Eigen::Vector3d>& contact_points,
                   std::vector<Eigen::Vector3d>& normals,
                   std::vector<Eigen::Vector3d>* barycentric_coordinates_ptr,
                   std::vector<Mesh::Triangle>* faces_indices_ptr)
{
    std::vector<std::vector<std::pair<Mesh::Triangle, Eigen::Vector3d>>> collision_infos =
      checkMeshAllCollision(vertices, mesh, tolerance);

    double tolerance_squared = tolerance * tolerance;

    double closest_point_distance_squared;
    Eigen::Vector3d closest_point_barycentric_coordinate;
    Eigen::Vector3d closest_point;
    Mesh::Triangle closest_point_triangle;

    double current_point_distance_squared;
    Eigen::Vector3d current_point_barycentric_coordinate;
    Eigen::Vector3d current_point;
    Mesh::Triangle current_point_triangle;
    for (std::size_t vertex_index = 0; vertex_index < vertices.size(); ++vertex_index)
    {
        const Eigen::Vector3d& vertex = vertices[vertex_index];
        closest_point_distance_squared = std::numeric_limits<double>::infinity();
        for (const auto& triangle_and_barycentric_coordinates : collision_infos[vertex_index])
        {
            boost::tie(current_point_triangle, current_point_barycentric_coordinate) =
              triangle_and_barycentric_coordinates;
            current_point =
              mesh.getTriangle(current_point_triangle) * current_point_barycentric_coordinate;
            current_point_distance_squared = (current_point - vertex).squaredNorm();
            if (current_point_distance_squared < closest_point_distance_squared)
            {
                closest_point = current_point;
                closest_point_barycentric_coordinate = current_point_barycentric_coordinate;
                closest_point_triangle = current_point_triangle;
                closest_point_distance_squared = current_point_distance_squared;
            }
        }

        // If there was at least one collision
        if (closest_point_distance_squared < tolerance_squared)
        {
            colliding_vertices_indices.push_back(vertex_index);
            contact_points.push_back(closest_point);
            normals.push_back(mesh.getTriangleNormal(closest_point_triangle));
            if (faces_indices_ptr)
            {
                faces_indices_ptr->push_back(closest_point_triangle);
            }
            if (barycentric_coordinates_ptr)
            {
                barycentric_coordinates_ptr->push_back(closest_point_barycentric_coordinate);
            }
        }
    }
}

std::vector<SelfCollisionInfo>
checkMeshSelfCollisions(const Mesh& mesh, double tolerance) noexcept
{
    std::vector<Eigen::Vector3d> vertices = mesh.getVertices();

    std::vector<std::vector<std::pair<Mesh::Triangle, Eigen::Vector3d>>> collisions_infos =
      checkMeshAllCollision(vertices, mesh, tolerance);

    std::vector<SelfCollisionInfo> result;
    Mesh::Triangle triangle;
    Eigen::Vector3d barycentric_coordinates;
    Eigen::Vector3d contact_point;
    SelfCollisionInfo self_collision_info;
    std::array<bool, 3> face_vertices_candidate;

    std::vector<std::vector<size_t>> registered;
    registered.resize(vertices.size());

    // Since we only do node-node contact, we pair each vertex in collision with
    // one other vertex. For each face, we make sure that the vertex is paired
    // with only one vertex of this face.  We will call the vertex of the
    // outside loop the colliding vertex.
    for (std::size_t vertex_index = 0; vertex_index < vertices.size(); ++vertex_index)
    {
        for (const auto& collision_info : collisions_infos[vertex_index])
        {
            boost::tie(triangle, barycentric_coordinates) = collision_info;
            // If the colliding vertex is part of the triangle or if it is on
            // the edge we don't count this as a collision.
            if (std::find(triangle.vertex_indices.begin(),
                          triangle.vertex_indices.end(),
                          vertex_index) != triangle.vertex_indices.end() ||
                barycentric_coordinates.x() == 0 || barycentric_coordinates.y() == 0 ||
                barycentric_coordinates.z() == 0)
            {
                continue;
            }

            // The colliding vertex is already paired with a vertex of this
            // triangle so we cannot count a collision with this triangle.
            if (isTriangleVertexIndexIn(triangle, registered[vertex_index]) < 3u)
            {
                continue;
            }

            // We try to find a vertex within the triangle that is not is
            // contact with a vertex adjacent to the colliding vertex. If we
            // were to link a vertex not satisfying this conditiong with the
            // colliding vertex. The linked vertex would be link to two vertex
            // that are part of the same triangle, the colliding vertex and its
            // adjacent vertex.
            std::fill(face_vertices_candidate.begin(), face_vertices_candidate.end(), true);
            for (std::size_t one_ring_vertex_index : mesh.getVerticesAroundVertexRange(vertex_index))
            {
                std::size_t triangle_vertex_index = isTriangleVertexIndexIn(triangle, registered[one_ring_vertex_index]);
                if (triangle_vertex_index < 3u)
                {
                    face_vertices_candidate[triangle_vertex_index] = false;
                }
            }

            // Among the candidate obtained in the previous loop. We take the
            // closest one.
            std::size_t closest_candidate_index = 3u;
            double closest_candidate_barycentric_component = 0;
            for (std::size_t candidate_index = 0; candidate_index < 3u; ++candidate_index)
            {
                if (!face_vertices_candidate[candidate_index])
                {
                    continue;
                }

                if (barycentric_coordinates[candidate_index] > closest_candidate_barycentric_component)
                {
                    closest_candidate_index = candidate_index;
                    closest_candidate_barycentric_component = barycentric_coordinates[candidate_index];
                }
            }

            //All the triangle vertices were already paired with one vertex of the one ring. Therefore, we don't pair this one.
            if (closest_candidate_index >= 3u)
            {
                continue;
            }

            contact_point = mesh.getTriangle(triangle) * barycentric_coordinates;
            self_collision_info.normal =
              (mesh.getVertex(vertex_index) - contact_point).normalized();
            
            std::size_t paired_vertex_index = triangle.vertex_indices[closest_candidate_index];
            barycentric_coordinates[closest_candidate_index] = 1.;
            barycentric_coordinates[(closest_candidate_index + 1) % 3] = 0.;
            barycentric_coordinates[(closest_candidate_index + 2) % 3] = 0.;

            // Keep track of which vertex has been paired with which.
            registered[paired_vertex_index].push_back(vertex_index);
            registered[vertex_index].push_back(paired_vertex_index);

            contact_point = mesh.getTriangle(triangle) * barycentric_coordinates;
            self_collision_info.barycentric_coordinates = barycentric_coordinates;            
            self_collision_info.face_indices = triangle.vertex_indices;
            self_collision_info.contact_point = contact_point;
#warning Suppose there is only one mesh simulated
            self_collision_info.vertex_index = vertex_index;
            result.push_back(std::move(self_collision_info));
              
            }
            
            
        }

    return result;
}

/**
 *  @return i where triangle.vertices_indices[i] is in vertices_indices. If there is no such i, returns triange.vertices_indices.size().
 */
std::size_t isTriangleVertexIndexIn(const Mesh::Triangle& triangle, const std::vector<size_t>& vertices_indices)
{
                auto triangle_vertex_index_it = std::find_first_of(
                    triangle.vertex_indices.begin(),
                    triangle.vertex_indices.end(),
                    vertices_indices.begin(),
                    vertices_indices.end());
                return std::distance(triangle.vertex_indices.begin(), triangle_vertex_index_it);
}

/**
 * @brief Find all collisions of the given vertices with the given mesh. The information on the
 * collisions of the vertex with index i are at the index i of the returned vector. The collision
 * information is stored in form of pairs of colliding triangle and barycentric coordinate of
 * contact point.
 */


std::vector<std::vector<std::pair<Mesh::Triangle, Eigen::Vector3d>>>
checkMeshAllCollision(const std::vector<Eigen::Vector3d>& vertices,
                      const Mesh& mesh,
                      double tolerance)
{
    std::vector<std::vector<Mesh::Triangle>> collision_to_check =
      getMeshPotentialTriangleCollision(vertices, mesh, 1.1 * tolerance);

    std::vector<std::vector<std::pair<Mesh::Triangle, Eigen::Vector3d>>> result(vertices.size());
    

    Eigen::Vector3d current_point;
    Eigen::Vector3d current_point_barycentric_coordinate;
    const double tolerance_squared = tolerance * tolerance;
    for (std::size_t vertex_index = 0; vertex_index < vertices.size(); ++vertex_index)
    {
        for (const Mesh::Triangle& triangle : collision_to_check[vertex_index])
        {
            const Eigen::Vector3d& vertex = vertices[vertex_index];
            closestPointToTriangle(vertex,
                                   mesh.getTriangle(triangle),
                                   tolerance,
                                   current_point,
                                   &current_point_barycentric_coordinate);
            
            if ((current_point - vertex).squaredNorm() < tolerance_squared)
            {
                result[vertex_index].push_back(
                  std::make_pair(triangle, current_point_barycentric_coordinate));
            }
        }
    }
    return result;
}

struct IndexBox
{
    CGAL::Box_intersection_d::Box_d<double, 3u> m_box;
    std::size_t m_index;
    using NT = double;
    using ID = std::size_t; 
    
    IndexBox(const IndexBox& other) = default;
    IndexBox(const CGAL::Bbox_3& box, std::size_t index) : m_box(box), m_index(index) {} 
    static int dimension() { return 3u; };
    ID id() const noexcept { return m_box.id(); }
    NT min_coord(int d) const noexcept { return m_box.min_coord(d); }
    NT max_coord(int d) const noexcept { return m_box.max_coord(d); }
    IndexBox& operator=(const IndexBox& other) = default;
};

/**
 * @brief The triangle at index i are the ones with whom the vertex with index i potentially
 * collide. This uses CGAL efficient bounding box intersection finder box_intersection_d.
 */
std::vector<std::vector<Mesh::Triangle>>
getMeshPotentialTriangleCollision(const std::vector<Eigen::Vector3d>& vertices,
                                  const Mesh& mesh,
                                  double tolerance)
{
    std::vector<IndexBox> mesh_boxes;
    for (std::size_t triangle_index = 0; triangle_index < mesh.getNumberOfTriangles(); ++triangle_index)
    {
        mesh_boxes.push_back(
          IndexBox(mesh.getTriangleBoundingBox(mesh.getTriangle(triangle_index)), triangle_index));
    }

    std::vector<IndexBox> vertices_boxes;
    for (std::size_t vertex_index = 0; vertex_index < vertices.size(); ++vertex_index)
    {
        vertices_boxes.push_back(
          IndexBox(getToleranceBoundingBox(vertices[vertex_index], tolerance), vertex_index));
    }

    std::vector<std::vector<Mesh::Triangle>> result(vertices.size());

    CGAL::box_intersection_d(
      mesh_boxes.begin(),
      mesh_boxes.end(),
      vertices_boxes.begin(),
      vertices_boxes.end(),
      [&result,&mesh](const IndexBox& triangle_box, const IndexBox& vertex_box) {
          result[vertex_box.m_index].push_back(mesh.getTriangle(triangle_box.m_index));
      });

    return result;
}

CGAL::Bbox_3
getToleranceBoundingBox(const Eigen::Vector3d& vertex, double tolerance)
{

    return CGAL::Bbox_3(vertex.x() - tolerance,
                        vertex.y() - tolerance,
                        vertex.z() - tolerance,
                        vertex.x() + tolerance,
                        vertex.y() + tolerance,
                        vertex.z() + tolerance);
}

bool
checkMeshCollision(const Eigen::Vector3d& pos,
                   const Mesh& mesh,
                   double tol2,
                   Eigen::Vector3d& contact_point,
                   Eigen::Vector3d& normal,
                   Eigen::Vector3d* barycentric_coordinates_ptr,
                   // Activate the self-collision exception mode
                   const size_t* vertex_index_ptr,
                   std::array<size_t, 3>* face_indices_ptr) noexcept
{

    // Find the closest face
    // loop that can be parallelized by OpenMP
    const Mesh::SurfaceMesh& m = mesh.getUnderlyingMesh();
    Mesh::SurfaceMesh::Face_range face_range = m.faces();

    double closest_dist2 = std::numeric_limits<double>::infinity();
    Eigen::Matrix3d closest_vertices;

    // TODO : test empty mesh
    Eigen::Matrix3d v;
    Eigen::Vector3d closest_on_face;
    Eigen::Vector3d bar;
    for (auto rb = face_range.begin(); rb != face_range.end(); ++rb)
    {
        // TODO : add a structure to mark some faces as not relevant
        // for the contact
        // clothcontact marks the vertices, and discard the triangle
        // if the 3 vertices are marked

        unsigned int i = 0;
        CGAL::Vertex_around_face_iterator<Mesh::SurfaceMesh> v_begin, v_end;
        bool self_contact = false;
        for (boost::tie(v_begin, v_end) = vertices_around_face(m.halfedge(*rb), m);
             v_begin != v_end;
             ++v_begin)
        {
            self_contact = self_contact || ((vertex_index_ptr) && (*vertex_index_ptr == *v_begin));
            v.col(i++) = mesh.getVertex(*v_begin);
        } // vertices
        assert(i == 3);

        // Skip this triangle to avoid the 1-ring
        if (self_contact)
        {
            continue;
        }

        // Test
        closestPointToTriangle(pos,
                               v,
                               tol2,
                               closest_on_face,
                               (barycentric_coordinates_ptr || vertex_index_ptr) ? &bar : nullptr);
        // Discard
        if ((vertex_index_ptr) && ((bar[0] == 0) || (bar[1] == 0) || (bar[2] == 0)))
        {
            continue;
        }

        const double dist2 = (pos - closest_on_face).squaredNorm();

        if (dist2 < closest_dist2)
        {
            contact_point = closest_on_face;
            closest_dist2 = dist2;
            closest_vertices = v;
            if (barycentric_coordinates_ptr)
            {
                *barycentric_coordinates_ptr = bar;
            }
            if (face_indices_ptr)
            {
                i = 0;
                for (boost::tie(v_begin, v_end) = vertices_around_face(m.halfedge(*rb), m);
                     v_begin != v_end;
                     ++v_begin)
                {
                    (*face_indices_ptr)[i++] = *v_begin;
                } // vertices
            }     // face_indices

        } // dist2

    } // faces

    const bool close = closest_dist2 < tol2;
    normal = (pos - contact_point).normalized();
    const bool inside =
      (!vertex_index_ptr) &&
      ((((closest_vertices.col(1) - closest_vertices.col(0))
           .cross(closest_vertices.col(2) - closest_vertices.col(0))) // face normal
          .dot(normal)) < 0);

    if ((!close) && (!inside))
    {
        return false;
    }

    if (inside)
    {
        normal *= -1;
    }
    return true;
}

std::vector<SelfCollisionInfo>
checkMeshSelfCollisions(const Eigen::Vector3d& pos,
                        const Mesh& mesh,
                        double tol2,
                        size_t vId) noexcept
{

    const Mesh::SurfaceMesh& m = mesh.getUnderlyingMesh();
    Mesh::SurfaceMesh::Face_range face_range = m.faces();

    std::vector<SelfCollisionInfo> result;

    for (auto rb = face_range.begin(); rb != face_range.end(); ++rb)
    {
        SelfCollisionInfo info;

        // Get the triangle
        Eigen::Matrix3d v;
        unsigned int i = 0;
        CGAL::Vertex_around_face_iterator<Mesh::SurfaceMesh> v_begin, v_end;
        bool ring = false;
        for (boost::tie(v_begin, v_end) = vertices_around_face(m.halfedge(*rb), m);
             v_begin != v_end;
             ++v_begin)
        {
            ring = ring || (vId == *v_begin);
            info.face_indices[i] = *v_begin;
            v.col(i++) = mesh.getVertex(*v_begin);
        } // vertices
        assert(i == 3);

        // Skip the 1 ring
        if (ring)
        {
            continue;
        }

        // Test
        closestPointToTriangle(pos, v, tol2, info.contact_point, &(info.barycentric_coordinates));
        // Discard
        if ((info.barycentric_coordinates[0] == 0) || (info.barycentric_coordinates[1] == 0) ||
            (info.barycentric_coordinates[2] == 0))
        {
            continue;
        }

        // Check if close
        const double dist2 = (pos - info.contact_point).squaredNorm();
        if (dist2 < tol2)
        {
            info.vertex_index = vId;
            info.normal = (pos - info.contact_point).normalized();
            result.push_back(info);
        }

    } // rb

    return result;
}

/**
 * @brief From Christer Ericson -- Real-Time Collision Detection (p141)
 *        (From an implementation by Gilles Daviet)
 *
 * @param p     Vertex to project
 * @param triangle Matrix whose column are the positions of the triangle vertices
 * @param tol2  Tolerance for the collision
 *
 * @return The barycentric coordinate of the closest point to p inside (abc)
 */
void
closestPointToTriangle(const Eigen::Vector3d& p,
                       const Eigen::Matrix3d& triangle,
                       double tol2,
                       Eigen::Vector3d& closest_point,
                       Eigen::Vector3d* barycentric_coordinates_ptr) noexcept
{
    const auto a = triangle.col(0);
    const auto b = triangle.col(1);
    const auto c = triangle.col(2);

    // Check if P in vertex region outside A
    const Eigen::Vector3d ap = p - a;
    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;

    // Early exit
    // Todo : tune tolerance

    if (ap.squaredNorm() - ab.squaredNorm() - ac.squaredNorm() > tol2)
    {
        closest_point = a;
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 1, 0, 0 };
        }
        return;
    }
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        closest_point = a;
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 1, 0, 0 };
        }
        return;
    }
    // Check if P in vertex region outside B
    const Eigen::Vector3d bp = p - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        closest_point = b;
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 0, 1, 0 };
        }
        return;
    }
    // Check if P in edge region of AB, if so return projection of P onto AB
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double v = d1 / (d1 - d3);
        closest_point = a + v * ab;
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 1 - v, v, 0 };
        }
        return;
    }
    // Check if P in vertex region outside C
    const Eigen::Vector3d cp = p - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        closest_point = c;
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 0, 0, 1 };
        }
        return;
    }
    // Check if P in edge region of AC, if so return projection of P onto AC
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double w = d2 / (d2 - d6);
        closest_point = a + w * ac;
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 1 - w, 0, w };
        }
        return;
    }
    // Check if P in edge region of BC, if so return projection of P onto BC
    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest_point = b + w * (c - b);
        if (barycentric_coordinates_ptr)
        {
            *barycentric_coordinates_ptr = Eigen::Vector3d{ 0, 1 - w, w };
        }
        return;
    }
    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    double denom = 1.0 / (va + vb + vc);
    double v = vb * denom;
    double w = vc * denom;
    closest_point = a + ab * v + ac * w;
    if (barycentric_coordinates_ptr)
    {
        *barycentric_coordinates_ptr = Eigen::Vector3d{ 1. - v - w, v, w };
    }
}

Eigen::Matrix3Xd
getLaplaceBeltramiDicretisation(const Eigen::Vector3d& middle_vertex,
                                const std::vector<Eigen::Vector3d>& ring_vertices)
{
    // There can't be only two triangle around the middle vertex.
    // It would mean that one of the triangle is flat, and therefore the
    // triangulation of the mesh would be wrong. Or the vertex would be on the
    // border and therefore the Laplace-Beltrami does not exist for this point.
    assert(ring_vertices.size() >= 3);

    Eigen::Matrix3Xd matrix = Eigen::Matrix3Xd::Zero(3, (ring_vertices.size() + 1) * 3);

    std::vector<double> coeff =
      getLaplaceBeltramiDiscretisationMeanValueCoefficients(middle_vertex, ring_vertices);

    Eigen::Matrix3d middle_vertex_block = Eigen::Matrix3d::Zero(3, 3);
    for (std::size_t i = 0; i < ring_vertices.size(); ++i)
    {
        matrix.block<3, 3>(0, (1 + i) * 3) = -coeff[i] * Eigen::Matrix3d::Identity();
        middle_vertex_block += coeff[i] * Eigen::Matrix3d::Identity();
    }
    matrix.block<3, 3>(0, 0) = middle_vertex_block;

    return matrix;
}


Eigen::MatrixXd
getReducedLaplaceBeltramiDicretisation(const Eigen::Vector3d& middle_vertex,
                                       const std::vector<Eigen::Vector3d>& ring_vertices)
{
    // There can't be only two triangle around the middle vertex.
    // It would mean that one of the triangle is flat, and therefore the
    // triangulation of the mesh would be wrong. Or the vertex would be on the
    // border and therefore the Laplace-Beltrami does not exist for this point.
    assert(ring_vertices.size() >= 3);

    Eigen::Matrix3Xd matrix = Eigen::Matrix3Xd::Zero(1, (ring_vertices.size() + 1) );

    std::vector<double> coeff =
      getLaplaceBeltramiDiscretisationMeanValueCoefficients(middle_vertex, ring_vertices);

    for (std::size_t i = 0; i < ring_vertices.size(); ++i)
    {
        matrix(0, (1 + i)) = -coeff[i];
        matrix(0, 0) += coeff[i] ;
    }

    return matrix;
}





std::vector<double>
getLaplaceBeltramiDiscretisationMeanValueCoefficients(
  const Eigen::Vector3d& middle_vertex,
  const std::vector<Eigen::Vector3d>& ring_vertices)
{
    // There can't be only two triangle around the middle vertex.
    // It would mean that one of the triangle is flat, and therefore the
    // triangulation of the mesh would be wrong. Or the vertex would be on the
    // border and therefore the Laplace-Beltrami does not exist for this point.
    assert(ring_vertices.size() >= 3);

    double clockwise_mean_value, counter_clockwise_mean_value;
    double distance;
    std::vector<double> result;
    for (std::size_t i = 0; i < ring_vertices.size(); ++i)
    {
        const Eigen::Vector3d& vertex = ring_vertices[i];
        const Eigen::Vector3d& clockwise_vertex = ring_vertices[(i + 1) % ring_vertices.size()];
        const Eigen::Vector3d& counter_clockwise_vertex =
          ring_vertices[(i + ring_vertices.size() - 1) % ring_vertices.size()];

        // We can't have a flat triangle around the middle vertex. If that
        // was the case, the triangulation of the mesh would be wrong or
        // the middle vertex would be on a border.
        assert(!areCollinear(vertex, clockwise_vertex, middle_vertex));
        assert(!areCollinear(vertex, middle_vertex, clockwise_vertex));

        distance = (vertex - middle_vertex).norm();

        clockwise_mean_value = getMeanValue(vertex, middle_vertex, clockwise_vertex);

        counter_clockwise_mean_value =
          getMeanValue(vertex, counter_clockwise_vertex, middle_vertex);

        result.push_back((clockwise_mean_value + counter_clockwise_mean_value) / distance);
    }
    return result;
}

std::vector<double>
getLaplaceBeltramiDiscretisationCotangentCoefficients(
  const Eigen::Vector3d& middle_vertex,
  const std::vector<Eigen::Vector3d>& ring_vertices)
{
    // There can't be only two triangle around the middle vertex.
    // It would mean that one of the triangle is flat, and therefore the
    // triangulation of the mesh would be wrong. Or the vertex would be on the
    // border and therefore the Laplace-Beltrami does not exist for this point.
    assert(ring_vertices.size() >= 3);

    double clockwise_cotangent, counter_clockwise_cotangent;
    std::vector<double> result;
    for (std::size_t i = 0; i < ring_vertices.size(); ++i)
    {
        const Eigen::Vector3d& vertex = ring_vertices[i];
        const Eigen::Vector3d& clockwise_vertex = ring_vertices[(i + 1) % ring_vertices.size()];
        const Eigen::Vector3d& counter_clockwise_vertex =
          ring_vertices[(i + ring_vertices.size() - 1) % ring_vertices.size()];

        // We can't have a flat triangle around the middle vertex. If that
        // was the case, the triangulation of the mesh would be wrong or
        // the middle vertex would be on a border.
        assert(!areCollinear(vertex, clockwise_vertex, middle_vertex));
        assert(!areCollinear(vertex, middle_vertex, clockwise_vertex));

        clockwise_cotangent = getCotangent(vertex, clockwise_vertex, middle_vertex);
        counter_clockwise_cotangent = getCotangent(vertex, counter_clockwise_vertex, middle_vertex);

        result.push_back(clockwise_cotangent + counter_clockwise_cotangent);
    }
    return result;
}

Eigen::Matrix3d
getChangeOfBasisIntoOrthonormalDirectBasis(const Eigen::Vector3d& vector)
{
    Eigen::Vector3d vector_normalized = vector.normalized();

    Eigen::Vector3d orthogonal_vector = vector_normalized.cross(Eigen::Vector3d(1, 0, 0));
    if (orthogonal_vector.norm() < PRECISION)
    {
        orthogonal_vector = vector_normalized.cross(Eigen::Vector3d(0, 1, 0));
    }
    orthogonal_vector.normalize();

    Eigen::Matrix3d result;

    result.col(0) = vector_normalized;
    result.col(1) = orthogonal_vector;
    result.col(2) = (vector_normalized.cross(orthogonal_vector)).normalized(); // Not needed
    return result;
}

double
getMeanValue(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3)
{
    double numerator = (1 - getCosine(v1, v2, v3));
    if (numerator == 0)
    {
        return 0;
    }
    return numerator / getSine(v1, v2, v3);
}

double
getSine(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3)
{
    Eigen::Vector3d edge1 = v1 - v2;
    Eigen::Vector3d edge2 = v3 - v2;
    return edge1.cross(edge2).norm() / (edge1.norm() * edge2.norm());
}

double
getCosine(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3)
{
    Eigen::Vector3d edge1 = v1 - v2;
    Eigen::Vector3d edge2 = v3 - v2;
    return edge1.dot(edge2) / (edge1.norm() * edge2.norm());
}

double
getCotangent(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3)
{
    Eigen::Vector3d edge1 = v1 - v2;
    Eigen::Vector3d edge2 = v3 - v2;
    return edge1.dot(edge2) / edge1.cross(edge2).norm();
}

bool
areCollinear(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3)
{
    return CGAL::collinear(CGAL::Point_3<EigenKernel>(v1),
                           CGAL::Point_3<EigenKernel>(v2),
                           CGAL::Point_3<EigenKernel>(v3));
}
