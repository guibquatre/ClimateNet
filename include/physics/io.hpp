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
#ifndef PHYSICS_IO_HPP
#define PHYSICS_IO_HPP


#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include "physics/SimulationBase.hpp"

#define OUTPUT_DIRECTORY_BASE (fs::path("data"))
#define OUTPUT_DIRECTORY_OFF (fs::path("output"))
#define OUTPUT_STAT_FILE (fs::path("stats.txt"))

namespace fs = boost::filesystem;

/**
 * @brief Create the directory containing the results of the simulation
 *        defined by the json file
 *        It has the name "dataXXX", with XXX the smallest uint available
 *        Also copy the conf file in the directory
 * @param scene_json_path  Conf file
 * @return The filepath of the directory
 */
fs::path initializeFileSystem(const fs::path& scene_json_path);


/**
 * @brief Get the filepath of the next off to dump
 * @param frame_number      Frame number to export
 * @param output_directory  Base output directory of the current sim
 * @return The filepath
 */
fs::path getOFFFilePath(std::size_t frame_number,
                        const fs::path& output_directory);

/**
 * @brief Get the filepath of the next obj to dump
 * @param frame_number      Frame number to export
 * @param output_directory  Base output directory of the current sim
 * @return The filepath
 */
fs::path getOBJFilePath(std::size_t frame_number,
                        const fs::path& output_directory);

/**
 * @brief Get the filepath where to write the simulation data
 * @param output_directory  Base output directory of the current sim
 * @return The filepath
 */
fs::path getStatsFilePath(const fs::path& output_directory);


// StatHelper
/**
 * @brief Write in an ofstream the stats on a vector
 * @param name
 * @param of
 * @param vec
 */
template <typename T>
void computeAndWriteStats(std::string name,
                          std::ofstream &of,
                          const std::vector<double> vec);
  
#include "physics/io.tpp"


#endif // PHYSICS_IO_HPP
