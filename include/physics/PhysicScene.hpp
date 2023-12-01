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
#ifndef SCENE_HPP
#define SCENE_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "cpp_utils.hpp"

#include "physics/PhysicMesh.hpp"
#include "physics/Obstacle.hpp"

#include "physics/AbstractConstraint.hpp"

#define COLLISION_CONSTRAINT_WEIGHT (1e2)


class PhysicScene
{
  /// @brief Iterator over the mesh list
  using MeshIterator = boost::transform_iterator<
    std::function<const PhysicMesh&(const std::shared_ptr<PhysicMesh>& mesh)>,
    std::vector<std::shared_ptr<PhysicMesh>>::const_iterator>;
  /// @brief Typedef
  using ObstaclePtr = std::unique_ptr<Obstacle>;

public:

  /// @brief Typedef
  using MeshIndexType = std::size_t;
  /// @brief Typedef
  using FrictionCoefficientTable = std::vector<std::vector<double>>;


  /**
   * @brief Copy constructor
   * @param other
   */
  PhysicScene(PhysicScene&& other) = default;
  /**
   * @brief Constructor
   * @param friction_coefficients Table of friction coefficients for the objects
   *                              Association made through the field 
   *                              "Material indentifier"
   */
  PhysicScene(const FrictionCoefficientTable& friction_coefficients);
  /**
   * @brief Constructor
   * @param friction_coefficients Table of friction coefficients for the objects
   *                              Association made through the field 
   *                              "Material indentifier"
   */
  PhysicScene(FrictionCoefficientTable&& friction_coefficients);

  /**
   * @brief Add a mesh to simulate
   * @param mesh
   */
    MeshIndexType addMesh(const PhysicMesh& mesh);
  /**
   * @brief Add a mesh to simulate
   * @param mesh
   */
    MeshIndexType addMesh(PhysicMesh&& mesh);
  /**
   * @brief Add a mesh to simulate
   * @param mesh_ptr
   */
    MeshIndexType addMesh(const std::shared_ptr<PhysicMesh>& mesh_ptr);

  /**
   * @brief Add an obstacle
   * @param obstacle
   */
  void addObstacle(std::unique_ptr<Obstacle>&& obstacle);

  /**
   * @brief Getter
   */
  const std::vector<ObstaclePtr>& getObstacles() noexcept;

  /**
   * @brief Update the scene internal with the information added by the user.
   *        This method has to be called before the scene is used 
   *        to get information.
   */
  void finalize();

  /**
   * @brief Add a Projective Dynamics user constraint
   * @param constraint_ptr
   */
  // We could use unique_ptr because the user doesn't have to own the
  // constraint this it is constant (If he wants to keep it, he just
  // has to make a copy). But, since the constraints are stored as
  // shared_ptr so that the simulation doesn't have to make a copy of the
  // users constaints, we take in shared_ptr to ease the user work.
  void addConstraint(const std::shared_ptr<AbstractConstraint>& constraint_ptr);

  /**
   * @brief Getter
   * @return The constraints
   */
  const std::vector<std::shared_ptr<AbstractConstraint>>& getConstraints();

  /**
   * @brief Compute and return the collisions
   * @param vertices
   * @return The collision informations
   */
  std::vector<CollisionInfo>
  getCollisionsInfo(const Eigen::VectorXd& positions) const;

  
  /**
   * @brief Compute and return the self collisions
   * @param vertices
   * @param tol2
   * @return The collision informations
   */
  std::vector<SelfCollisionInfo>
  getSelfCollisionsInfo(double tol2) const;

  

  /**
   * @brief Return the collision info for each except for the vertices
   *         whose index are in the \a exceptions collection. 
   * @param vertices
   * @param exceptions  must be sorted
   */
  std::vector<CollisionInfo> getCollisionsInfoExceptFor(
    const Eigen::VectorXd& positions,
    const std::vector<std::size_t> exceptions) const;

  /**
   * @brief Getter
   * @return The friction coefficient table
   */
  const FrictionCoefficientTable& getFrictionCoefficientTable() const noexcept;

  /**
   * @brief Getter
   * @return The mass matrix
   */
  const SparseMatrix& getMassMatrix() const;
  /**
   * @brief Getter
   * @return The inverse of the mass matrix
   */
  const SparseMatrix& getInverseMassMatrix() const;

  /**
   * @brief Getter
   * @return The mass of a vertex
   */
  double getVertexMass(size_t vId) const noexcept;

  /**
   * @brief Getter
   * @return The mass matrix
   */
  const SparseMatrix& getReducedMassMatrix() const;
  /**
   * @brief Getter
   * @return The inverse of the mass matrix
   */
  const SparseMatrix& getReducedInverseMassMatrix() const;
  
  /**
   * @brief Advance of \a step the time of time-dependent objects.
   */
  void updateTimeDependentObjects(double step);
  /**
   * @brief Getter on the generalized positions *of all the meshes*
   * @return Positions
   */
  const Eigen::VectorXd& getGeneralizedPositions();
  /**
   * @brief Update the internal concatenation of the positions
   *        To be used if the meshes positions have been modified
   *        directly without using setGeneralizedPositions
   */
  void updateGeneralizedPositions();
  /**
   * @brief Setter
   * @param q  The new positions
   */
  void setGeneralizedPositions(const Eigen::VectorXd& q);


  /**
   * @brief Getter on the generalized speeds *of all the meshes*
   * @return Speeds
   */
  const Eigen::VectorXd& getGeneralizedSpeeds();
  /**
   * @brief Update the internal concatenation of the speeds
   *        To be used if the meshes speeds have been modified
   *        directly without using setGeneralizedSpeeds
   */
  void updateGeneralizedSpeeds();
  /**
   * @brief Setter
   * @param speed  The new speeds
   */
  void setGeneralizedSpeeds(const Eigen::VectorXd& speed);

  /**
   * @brief Get the generalized index of a vertex
   * @param mesh_index  
   * @param vertex_index  Index of the vertex in the mesh_index
   * @return Index in the concatenated vector
   */
  std::size_t getMeshVertexSceneIndex(MeshIndexType mesh_index,
                                      std::size_t vertex_index) const;

  /**
   * @brief Getter
   * @param index  Mesh index
   * @return Mesh pointer
   */
  std::shared_ptr<PhysicMesh> getMesh(std::size_t index);

  std::shared_ptr<const PhysicMesh> getMesh(std::size_t index) const;

  /**
   * @brief Getter
   * @return Number of vertices
   */
  std::size_t getNumberOfVertices() const;
  
  /**
   * @brief Getter
   * @return Number of Dofs
   */
  std::size_t getNumberOfDofs() const;

private:

  /// @brief Helper - Compute the mass matrix and the inverse mass matrix
  void initializeMassMatrices();

  /**
   * @brief Get the mesh corresponding to a generalized index
   * @param vertex_index  Generalized index
   * @param Mesh ptr
   */
  std::shared_ptr<PhysicMesh> getContainingMesh(std::size_t vertex_index) const;

  /**
   * @brief Compute the vertex indices offset
   * @param mesh_index
   * @return The offset
   */
  std::size_t getNumberOfVerticesUntil(std::size_t mesh_index) const;

  /**
   * @brief Helper
   * @return
   */
  std::function<std::size_t(const PhysicMesh&)>
  getMeshNumberOfVerticesFunctionObject() const;

  /**
   * @brief Iterator on the meshes
   * @return Begin
   */
  MeshIterator getMeshIteratorBegin() const;
  /**
   * @brief Iterator on the meshes
   * @return End
   */
  MeshIterator getMeshIteratorEnd() const;

private:
  /// @brief ...
  std::vector<std::shared_ptr<PhysicMesh>> m_meshes;
  /// @brief ...
  std::vector<ObstaclePtr> m_obstacles;
  /// @brief ...
  std::vector<std::shared_ptr<AbstractConstraint>> m_user_constraints;
  /// @brief ...
  std::vector<std::shared_ptr<AbstractConstraint>> m_constraints;

  /// @brief ...
  FrictionCoefficientTable m_friction_coefficients;

  /// @brief ...
  SparseMatrix m_mass_matrix;
  /// @brief ...
  SparseMatrix m_inverse_mass_matrix;

  //TODO : avoid doubling the matrices
  /// @brief ...
  SparseMatrix m_reduced_mass_matrix;
  /// @brief ...
  SparseMatrix m_reduced_inverse_mass_matrix;

  /// @brief ...
  Eigen::VectorXd m_generalized_positions;
  /// @brief ...
  Eigen::VectorXd m_generalized_speeds;
};

#endif
