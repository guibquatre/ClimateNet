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
#include "jsonHelper.hpp"

rapidjson::Document
getJSONDocumentFromFilename(const std::string& filename)
{
  std::FILE* file = std::fopen(filename.c_str(), "r");
  if (file == nullptr)
  {
    throw std::invalid_argument(std::string("Couldn't open file ") + filename);
  }

  constexpr std::size_t buffer_size = 1024;
  char buffer[buffer_size];
  // TODO: If an exception is thrown the file might not be closed correctly.
  // Either create a wrapper around std::FILE with an appropiate destructor
  // or used an appropriate library.
  rapidjson::Document document;
  rapidjson::FileReadStream is =
    rapidjson::FileReadStream(file, buffer, buffer_size);
  document.ParseStream(is);
  std::fclose(file);
  return document;
}
