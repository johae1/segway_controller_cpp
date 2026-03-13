cmake_minimum_required(VERSION 3.8)
project(segway_controller_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(controller_node_cpp src/controller_node.cpp)
ament_target_dependencies(controller_node_cpp rclcpp std_msgs geometry_msgs)
target_compile_features(controller_node_cpp PUBLIC cxx_std_17)

add_executable(state_vector_node_cpp src/state_vector_node.cpp)
ament_target_dependencies(state_vector_node_cpp rclcpp std_msgs sensor_msgs nav_msgs)
target_compile_features(state_vector_node_cpp PUBLIC cxx_std_17)

install(TARGETS
  state_vector_node_cpp
  controller_node_cpp
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
