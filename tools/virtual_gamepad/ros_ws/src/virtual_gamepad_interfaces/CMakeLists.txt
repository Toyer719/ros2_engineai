cmake_minimum_required(VERSION 3.8)
project(virtual_gamepad_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/WalkTo.action"
  "action/Stand.action"
  "action/Lift.action"
  "action/Pivot.action"
  "action/Depose.action"
  DEPENDENCIES action_msgs
)

ament_package()
