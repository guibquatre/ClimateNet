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
#include "simulation_building.hpp"

#include <iostream>
#include <fstream>

#include "jsonHelper.hpp"

#include "physics/SimulationSpeed.hpp"
#include "physics/SimulationFrictionEstimation.hpp"

SimulationParameters
getSimulationParametersFromJSONFileName(const char* filename)
{
  return getSimulationParametersFromJSONDocument(
    getJSONDocumentFromFilename(filename));
}

SimulationParameters
getSimulationParametersFromJSONDocument(const rapidjson::Document& document)
{
  // Read the parameters
  SimulationParameters params;

  const rapidjson::Value& parametersValue =
    jsonGetValue(document, "Parameters");

  params.max_output_frame = jsonRequireUint(parametersValue, "Frame number");
  params.local_global_iterations = jsonRequireUint(parametersValue, "LocalGlobal");
  params.time_step = jsonRequireDouble(parametersValue, "Timestep");
  params.sim_steps_per_frame = jsonRequireUint(parametersValue, "Steps per frame");
  params.self_collision = jsonRequireBool(parametersValue, "Self-collision");
  if (params.self_collision)
  {
    params.self_collision_tolerance =
      jsonRequireDouble(parametersValue, "Self-collision tolerance");
  }
  else
  {
    params.self_collision_tolerance = 0.;
  }
  const rapidjson::Value& gravity_value
    = jsonRequireArrayCheck(parametersValue, "Gravity");
  for (size_t cmp = 0u; cmp < 3u; ++cmp)
  {
    params.gravity[cmp] = gravity_value[cmp].GetDouble();
  }

  // BAD
  try
  {
    params.air_damping = jsonRequireDouble(parametersValue, "Air damping");
  }
  catch (std::invalid_argument e)
  {
    params.air_damping = 0.;
  }

  return params;
}

std::unique_ptr<SimulationBase>
getSimulation(PhysicScene& scene,
              const SimulationParameters& params)
{

  // Gravity
  const Eigen::Vector3d &g = params.gravity;
  std::function<Eigen::Vector3d(const Eigen::Matrix3d&,
                                const Eigen::Vector3d&,
                                double)>
    gravity_functor = [g](const Eigen::Matrix3d& mass_matrix,
                          const Eigen::Vector3d&,
                          double) {return mass_matrix * g;};

    return std::make_unique<SimulationFrictionEstimation>(
      scene, params.time_step, params.local_global_iterations,
      gravity_functor, params.air_damping,
      params.self_collision, params.self_collision_tolerance);
}

