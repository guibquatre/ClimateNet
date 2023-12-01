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
#include "physics/io.hpp"
#include <numeric>

fs::path
initializeFileSystem(const fs::path& scene_json_path)
{

  // Create the directory
  unsigned int directory_number = 0;
  fs::path output_directory;
  do {
    output_directory =
      fs::path(OUTPUT_DIRECTORY_BASE.string()
               + std::to_string(directory_number));
    ++directory_number;
  } while (fs::is_directory(output_directory));
  fs::create_directories(output_directory);

  // Copy the configuration file inside
  fs::copy_file(scene_json_path,
                output_directory / scene_json_path.filename());

  // Create the directory for the off
  fs::create_directories(output_directory / OUTPUT_DIRECTORY_OFF);

  return output_directory;
}



fs::path
getOFFFilePath(std::size_t frame_number,
               const fs::path& output_directory)
{
    std::stringstream number_stream;
    number_stream << std::setw(6) << std::setfill('0') << frame_number;
    return output_directory / OUTPUT_DIRECTORY_OFF
      /  ("/out_" + number_stream.str() + ".off");
}

fs::path
getOBJFilePath(std::size_t frame_number,
               const fs::path& output_directory)
{
    std::stringstream number_stream;
    number_stream << std::setw(6) << std::setfill('0') << frame_number;
    return output_directory / OUTPUT_DIRECTORY_OFF
      /  ("/out_" + number_stream.str() + ".obj");
}

fs::path
getStatsFilePath(const fs::path& output_directory)
{
  return (output_directory / OUTPUT_STAT_FILE);
}


