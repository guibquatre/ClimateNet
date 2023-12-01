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
#ifndef SIMULATION_TEST_HPP
#define SIMULATION_TEST_HPP

#include "physics/SimulationSpeed.hpp"

/**
 * @brief Class implementing our modified speed PD for estimating
 *        the frictional contact
 */
class SimulationFrictionEstimation : public SimulationSpeed
{
  public:
    /**
     * @brief Constructor
     * @param scene                  Set up to simulate
     * @param time_step              Simulation timestep
     * @param iteration_number       Number of iterations
     * @param external_forces        Function pointer to compute the external forces
     * @param handle_self_collision
     */
    SimulationFrictionEstimation(PhysicScene& scene,
                                 double time_step,
                                 std::size_t iteration_number,
                                 const ExternalForce& external_force,
                                 double air_damping,
                                 bool handle_self_collision,
                                 double self_collision_tolerance);

    /**
     * @brief Write all the stats in the given ofstream
     */
    virtual void exportStats(std::ofstream& of) const noexcept override;

  private:
    virtual void updateScene() noexcept override;

    virtual void updateCollision() noexcept override;

    virtual void buildIteration() noexcept override;

    std::size_t getSelfCollisionContactPointIndex(
      const SelfCollisionInfo& self_collision_info) const noexcept;
    /**
     * Computes the order in which the computations of repulsive forces should
     * be done for better stability. 
     * @see m_collision_computation_order
     */
    void computeCollisionComputationOrder() noexcept;
    /**
     * Compute the self-collision graph. The graph vertices are the vertices
     * of the mesh, the edges are the self-collision between vertices.
     * @see m_self_collision_graph
     */
    void computeSelfCollisionGraph() noexcept;
    /**
     * Fills the collision computation order.
     *
     * The vertices are parcoured through
     * breadth first search following the collision graph starting from the
     * given vertices. In the directed acyclic graph produced by the breadth first search, the
     * collisions associated with the edges at the beginning of the graph are
     * put into the given computation level. The collision associated to the
     * edges coming after these edges are put on the next computation level.
     * This is done until the collisions associated to the edges attained by the
     * breadth first search have been added to a computation level.
     *
     * Before a call to this function is made, there should have been a called
     * to computeSelfCollisionGraph since the last collisions detection.
     *
     * @param starting_vertices The vertices at which the breadth first search
     *                          will be started.
     * @param visited A vector of boolean indicating which collision edge has
     *                already been visited. If `visited[i]` is true, the
     *                collision with index `i` has been visited. The index used
     *                is the index of the collision within
     *                m_self_collisions_infos.
     * @param initial_computation_index The compution level of the first edges
     *                                  encountered in the breadth first search.
     * @see computeSelfCollisionGraph
     * @see computeCollisionComputationOrder
     */
    void fillCollisionComputationOrder(
      const std::vector<std::size_t>& starting_vertices,
      std::vector<bool>& visited,
      std::size_t initial_computation_index) noexcept;
    /**
     * Returns if a vertex has been visited during a breadth first search of the
     * self-collision graph from its visited edges. A vertex is considered to
     * have been visited if all of its edges have been visited.
     * @param vertex_index The index of the vertex within the simulated mesh.
     * @param visited A vector of boolean indicating which collision edge has
     *                already been visited. If `visited[i]` is true, the
     *                collision with index `i` has been visited. The index used
     *                is the index of the collision within
     *                m_self_collisions_infos.
     * @see fillCollisionComputationOrder               
     * @see computeSelfCollisionGraph
     */
    bool vertexIsVisited(std::size_t vertex_index, const std::vector<bool>& visited) const noexcept;

    virtual void computeRhs(
      const Eigen::VectorXd& current_next_speed,
      const Eigen::VectorXd& next_speed_under_no_constraints) noexcept override;

    void updateAlartCurnierNorms() noexcept;
    void updateCollisionsLocalVelocities() noexcept;
    std::vector<std::size_t> getSelfCollisionVerticesIndices() const noexcept;
    std::vector<std::size_t> getCollisionVerticesIndices() const noexcept;
    std::vector<double> getSelfCollisionFrictionCoefficients() const noexcept;
    std::vector<double> getCollisionFrictionCoefficients() const noexcept;
    void updateLocalVelocities(
            const std::vector<std::size_t>& vertices_indices,
            const std::vector<Eigen::Matrix3d>& local_basis,
            std::vector<Eigen::Vector3d>& local_velocities) noexcept;

    virtual void updateIterationData() noexcept override;


    std::vector<Eigen::Matrix3d> m_local_contact_basis;

    /**
     * Represent an edge in the self-collision graph adjacency list.
     * @see m_self_collision_graph
     */
    struct SelfCollisionGraphNeighboor
    {
        /**
         * The index of vertex adjacent through this edge.
         */
        std::size_t index;
        /**
         * The index of the self-collision represented by this index.
         */
        std::size_t collision_index;
    };
    /**
     * An adjacency list representing the self-collision graph. The vertices of
     * this graph are the vertices of the mesh. The edges are the self-collision
     * between the vertices.
     * @see SelfCollisionGraphNeighboor
     * @see computeSelfCollisionGraph
     */
    std::vector<std::vector<SelfCollisionGraphNeighboor>> m_self_collision_graph;
    std::vector<Eigen::Matrix3d> m_local_self_contact_basis;
    /**
     * The order in which the computation of repulsive force should be made.
     * The first element of this vector contains the indices of the
     * self-collision whose repulsion force should be computed first, the second
     * element those whose repulsion force that should be computed in second. So
     * on and so forth. Its value can be computed through
     * computeSelfCollisionGraph. The self-collision on a same level can be
     * safelly computed in parallel.
     */
    std::vector<std::vector<std::size_t>> m_collision_computation_order;

    /// @brief (IO)
    std::vector<double> m_rhs_base_times;
    /// @brief (IO)
    std::vector<double> m_friction_times;
    /// @brief (IO)
    std::vector<double> m_self_friction_times;
    std::vector<double> m_self_collision_ordering_times;

    /// Better Row or column ?
    // column is better to write in
    // row is better to add to rhs at the end
    using ForceType = Eigen::Matrix<double, 3, -1, Eigen::RowMajor>;

    unsigned int m_current_index;
    unsigned int m_next_index;
    /// @brief Storage needed to handle self friction
    ForceType m_contact_forces[2];
    /// @brief Storage needed to handle self friction
    ForceType m_self_contact_forces[2];
    /// @brief Storage needed to handle self friction
    ForceType m_self_contact_repercusion_forces[2];
    /// @brief Storage needed to handle self friction
    Eigen::Matrix3Xd m_alpha[2];

    std::vector<Eigen::Vector3d> m_collisions_local_forces;
    std::vector<Eigen::Vector3d> m_collisions_local_velocities;
    std::vector<double> m_collisions_friction_coefficients;
    std::vector<double> m_alart_curnier_norms;
    std::vector<Eigen::Vector3d> m_self_collisions_local_forces;
    std::vector<Eigen::Vector3d> m_self_collisions_local_velocities;
    std::vector<double> m_self_collisions_friction_coefficients;
    std::vector<double> m_self_alart_curnier_norms;
    std::vector<double> m_self_normal_alart_curnier;
    std::vector<double> m_normal_alart_curnier;
    std::vector<double> m_self_tangential_alart_curnier_norms;
    std::vector<double> m_tangential_alart_curnier_norms;

    Eigen::Matrix3Xd m_remember_self_contact_forces;

};

#endif // SimulationSpeed
