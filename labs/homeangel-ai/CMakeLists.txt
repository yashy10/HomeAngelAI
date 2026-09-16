cmake_minimum_required(VERSION 3.16)
project(homeangel-ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NEAT_APPS_ROOT "/neat-resources/apps-src" CACHE PATH "Packaged SiMa Neat apps source")
set(SimaNeat_DIR "/opt/toolchain/aarch64/modalix/usr/lib/cmake/SimaNeat" CACHE PATH "SimaNeat CMake package")
list(APPEND CMAKE_PREFIX_PATH "/opt/toolchain/aarch64/modalix/usr")

include("${NEAT_APPS_ROOT}/cmake/ExampleModule.cmake")
_sima_neat_apps_ensure_neat_target("${NEAT_APPS_ROOT}")
find_package(nlohmann_json REQUIRED)

set(HOMEANGEL_OPENCV_INCLUDE_DIR "/opt/toolchain/aarch64/modalix/usr/include/opencv4")
set(HOMEANGEL_OPENCV_LIBRARY_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu")

add_library(homeangel_config STATIC
  "${NEAT_APPS_ROOT}/support/runtime/config_utils.cpp"
)
target_include_directories(homeangel_config
  PUBLIC
    "${NEAT_APPS_ROOT}"
)

add_library(homeangel_tracker STATIC
  src/utils/tracker.cpp
)
target_include_directories(homeangel_tracker
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(homeangel_tracker
  PUBLIC
    SimaNeatApps::sima_neat
)

add_executable(homeangel-ai
  src/main.cpp
)
target_include_directories(homeangel-ai
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${NEAT_APPS_ROOT}"
    "${HOMEANGEL_OPENCV_INCLUDE_DIR}"
)
target_compile_definitions(homeangel-ai
  PRIVATE
    SIMANEAT_APPS_EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(homeangel-ai
  PRIVATE
    homeangel_tracker
    homeangel_config
    SimaNeatApps::sima_neat
    nlohmann_json::nlohmann_json
    opencv_imgcodecs
    opencv_imgproc
    opencv_core
)
target_link_directories(homeangel-ai
  PRIVATE
    "${HOMEANGEL_OPENCV_LIBRARY_DIR}"
)
set_target_properties(homeangel-ai PROPERTIES
  BUILD_RPATH "/opt/toolchain/aarch64/modalix/usr/lib;/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
  INSTALL_RPATH "/opt/toolchain/aarch64/modalix/usr/lib;/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
)
