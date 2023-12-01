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
#ifndef SIMULATION_BUILDING_HPP
#define SIMULATION_BUILDING_HPP

#include <memory>
#include "physics/PhysicScene.hpp"
#include "physics/SimulationBase.hpp"
#include "rapidjson/document.h"

/** @file
 * Functions for retrieving simulation parameters from a JSON file and building
 * a simulation from these parameters. For documentation on the JSON file see
 * `Configuration.md` in the repository.
 */

struct SimulationParameters
{
  size_t max_output_frame;
  size_t local_global_iterations;
  double time_step;
  std::string simulation_type;
  size_t sim_steps_per_frame;
  bool self_collision;
  double self_collision_tolerance;
  Eigen::Vector3d gravity;
  double air_damping;
};

/**
 * @brief Reads json file and load the simulation parameters
 * @param filename  Json filepath
 * @return SimulationParameters
 */
SimulationParameters
getSimulationParametersFromJSONFileName(const char* filename);

SimulationParameters
getSimulationParametersFromJSONDocument(const rapidjson::Document& document);

/**
 * @brief Create the simulation
 *        from a physic scene and simulation parameters. The simulation is
 *        returned as a `std::unique_ptr<SimulationBase>`. The class of the
 *        object pointed by the returned pointer depends on the value of
 *        `params.simulation_type`
 *        - "speed" : `SimulationSpeed`;
 *        - "friction" : `SimulationFrictionEstimation`.
 * @param scene
 * @param params
 * @return Simulation object
 */
std::unique_ptr<SimulationBase>
getSimulation(PhysicScene& scene, const SimulationParameters& params);

#endif // SIMULATION_BUILDING_HPP
