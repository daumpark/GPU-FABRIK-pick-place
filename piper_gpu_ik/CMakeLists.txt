cmake_minimum_required(VERSION 3.5)
project(piper_gpu_ik)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(PYTHON_PKG piper_gpu_ik_py)
ament_python_install_package(${PYTHON_PKG})

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/BatchIk.srv"
  DEPENDENCIES geometry_msgs
)

# ros2 run 으로 확장자 없이 실행 가능하도록 파일명 리네임하여 설치
install(PROGRAMS
  ${PYTHON_PKG}/gpu_fabrik_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gpu_fabrik_server
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
