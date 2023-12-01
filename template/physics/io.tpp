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
#ifndef PHYSICS_IO_TPP
#define PHYSICS_IO_TPP

template <typename T>
void computeAndWriteStats(std::string name,
                          std::ofstream &of,
                          const std::vector<T> vec)
{
  
  of << name << std::endl;

  if (vec.size() == 0)
  {
    of << "\t(No data)" << std::endl;
  }
  else if (vec.size() == 1)
  {
    of << "\t Value : " << vec[0] << std::endl;
  }
  else
  {
    const T min_value =
      std::accumulate(vec.begin(), vec.end(),
                      std::numeric_limits<double>::max(),
                      [](double a, double b) { return (std::min(a, b)); } );
    const T mean_value =
      std::accumulate(vec.begin(), vec.end(), 0.) / vec.size();
    const T max_value =
      std::accumulate(vec.begin(), vec.end(),
                      std::numeric_limits<double>::min(), 
                      [](double a, double b) { return (std::max(a, b)); } );

  
    of << "\t Number      : " << vec.size() << std::endl  
       << "\t Min value   : " << min_value << std::endl
       << "\t Mean value  : " << mean_value << std::endl
       << "\t Max value   : " << max_value << std::endl;
  }
  of << std::endl
     << "-------------------"
     << std::endl
     << std::endl;

}


#endif // PHYSICS_IO_TPP
