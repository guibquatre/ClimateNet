#==========================================
#Project options.
#==========================================
cmake_minimum_required(VERSION 3.9)

option(PRINT_ALART_CURNIER_ERROR OFF "Print alart curnier error to standard output")


#==========================================
#Project name.
#==========================================

project(FrictionProjectiveDynamics)


#==========================================
#Building options.
#==========================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


#==========================================
#Building mode handling.
#==========================================
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE DEBUG)
    set(CMAKE_CONFIGURATION_TYPES Debug)
else()
    set(CMAKE_CONFIGURATION_TYPES Release)
endif()
message("Building mode: " ${CMAKE_BUILD_TYPE})

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  
#==========================================
#Libraries.
#==========================================

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
#cmake_policy(SET CMP0077 NEW)

# Doxygen
find_package(Doxygen)
if (DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif(DOXYGEN_FOUND)

# CGAL
set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS true)
find_package(CGAL COMPONENTS Core)
include(${CGAL_USE_FILE})

# Boost
find_package(Boost COMPONENTS filesystem REQUIRED)

# Tiny Obj Loader
set(TOL_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/extern/tinyObjLoader"
  CACHE PATH "TOL")
include_directories(${TOL_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 3.2.92 REQUIRED)
if (EIGEN3_FOUND)
   include_directories(${EIGEN3_INCLUDE_DIR})
endif()

#OpenMP
find_package(OpenMP)
if(PROFILING)
    set (CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed -lprofiler ${CMAKE_EXE_LINKER_FLAGS}")
endif()

# Rapidjson
if (NOT RAPIDJSON_INCLUDEDIR)
  set(RAPIDJSON_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/rapidjson/include")
endif()
find_package(rapidjson REQUIRED)
include_directories("${RAPIDJSON_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RAPIDJSON_CXX_FLAGS}")

#==============================================
#Project sources.
#==============================================

set(EXECUTABLE_NAME ProjectiveFriction)

file(
    GLOB_RECURSE
    SOURCE_FILES
    src/*.cpp
    )

add_to_cached_list(CGAL_EXECUTABLE_TARGETS ${EXECUTABLE_NAME})

add_executable(${EXECUTABLE_NAME} ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)
target_include_directories(${EXECUTABLE_NAME} PRIVATE template)
if (${PRINT_ALART_CURNIER_ERROR})
  target_compile_definitions(${EXECUTABLE_NAME}
    PRIVATE PRINT_ALART_CURNIER_ERROR)
endif()


#==============================================
#Linking with libraries.
#==============================================
# CGAL & Boost
target_link_libraries(${EXECUTABLE_NAME}
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY})
if (NOT PROFILING AND OpenMP_CXX_FOUND)
  target_link_libraries(${EXECUTABLE_NAME} OpenMP::OpenMP_CXX)
endif()

message("${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
