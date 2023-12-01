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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <list>

#include <Eigen/Core>

#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Point_3.h>

#include "geometry/io.hpp"
#include "jsonHelper.hpp"
#include "physics/io.hpp"

#include "log_building.hpp"
#include "scene_building.hpp"
#include "simulation_building.hpp"

int
main(int argc, char* argv[])
{
  // Input
  if (argc <= 1)
  {
    std::cerr << "Not enough arguments - 1 argument expected" << std::endl;
    std::cerr << " (1) Path of the json configuration file " << std::endl;
    return EXIT_FAILURE;
  }
  const fs::path scene_json_path(argv[1]);

  std::cout << "# Initializing." << std::endl;

  // Output
  const fs::path output_directory = initializeFileSystem(scene_json_path);

  rapidjson::Document configuration_document =
    getJSONDocumentFromFilename(scene_json_path.string());

  // Scene
  PhysicScene scene = getSceneFromJSONDocument(configuration_document);
  std::shared_ptr<PhysicMesh> mesh_ptr = scene.getMesh(0);

  // Parameters
  const SimulationParameters params =
    getSimulationParametersFromJSONDocument(configuration_document);

  // Simulator
  auto simulation = getSimulation(scene, params);

  StepCallBacks log_callback =
    getLogCallBacksFromJSONDocument(configuration_document);

  const size_t max_number_steps =
    params.max_output_frame * params.sim_steps_per_frame;
  std::cout << "# Initialization done. Starting simulation, "
            << max_number_steps << " timesteps" << std::endl;

  // Progress
  std::list<int> milestones; // in percentage
  const int dm = 5;
  for (int i = dm; i <= 100; i += dm) {
    milestones.push_back(i);
  }

  for (std::size_t i = 0; i < max_number_steps; ++i)
  {

    // Compute
    std::cout << "Time Step: " << i << std::endl;
    simulation->step();
    log_callback(i, scene);

    // Output
    if (i % params.sim_steps_per_frame == 0)
    {
      mesh_ptr->updateUnderlyingMesh();

      std::cout << "Saving Mesh" << std::endl;
      std::ofstream of(
        getOBJFilePath(i / params.sim_steps_per_frame, output_directory)
          .c_str());
      // of << mesh_ptr->getUnderlyingMesh(); // This lines write the mesh as an
      // OFF file
      mesh_ptr->writeObj(of);
      of.close();
    }

    if ( (!milestones.empty())
         && (i >= milestones.front() * (max_number_steps - 1) / 100.) ) {
      std::cerr <<  milestones.front() << "% done" << std::endl;
      milestones.pop_front();
    }
  }

  std::cout << "# Simulation done. Writing stats." << std::endl;

  std::ofstream of(getStatsFilePath(output_directory).c_str());
  simulation->exportStats(of);

  std::cout << "# Writing done. " << std::endl;

  return EXIT_SUCCESS;
}
