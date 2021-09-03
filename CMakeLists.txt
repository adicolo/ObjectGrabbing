cmake_minimum_required(VERSION 2.8)

project(libfranka-examples CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/../cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Poco REQUIRED COMPONENTS Foundation)

find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(examples_common STATIC
  examples_common.cpp
)

target_link_libraries(examples_common PUBLIC Franka::Franka Eigen3::Eigen3)

set(EXAMPLES
  point_to_point_joint_adrien
  point_to_point_cartesian_adrien
  via_points_joint_adrien
  via_points_joint_adrien_copy
  via_points_cartesian_adrien
  via_points_cartesian_adrien_copy
  via_points_cartesian_adrien_calibration
  cartesian_impedance_control
  communication_test
  echo_robot_state
  force_control
  generate_cartesian_pose_motion
  generate_cartesian_velocity_motion
  generate_consecutive_motions
  generate_elbow_motion
  generate_joint_position_motion
  generate_joint_velocity_motion
  grasp_object
  joint_impedance_control
  joint_point_to_point_motion
  motion_with_control
  print_joint_poses
  vacuum_object
)

# SET(OPENCV_FLAGS "`pkg-config opencv --cflags --libs` -std=c++11 -lrealsense2")
# add_definitions(${OPENCV_FLAGS})

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} Franka::Franka examples_common Eigen3::Eigen3)
endforeach()

add_executable(via_points_cartesian_adrien_calibration_test via_points_cartesian_adrien_calibration_test.cpp)
target_link_libraries(via_points_cartesian_adrien_calibration_test Franka::Franka examples_common Eigen3::Eigen3 ${OpenCV_LIBS} ${REALSENSE2_FOUND})

add_executable(main main.cpp)
target_link_libraries(main Franka::Franka examples_common Eigen3::Eigen3 ${OpenCV_LIBS} ${REALSENSE2_FOUND})

target_link_libraries(joint_impedance_control Threads::Threads)
target_link_libraries(motion_with_control Poco::Foundation)

include(GNUInstallDirs)
install(TARGETS ${EXAMPLES}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
