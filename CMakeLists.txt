cmake_minimum_required(VERSION 3.0.2)
project(box_yolo_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  detection_msgs
  std_srvs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES yolov5_ros
#  CATKIN_DEPENDS rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
