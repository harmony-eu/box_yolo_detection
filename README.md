## Usage
The nodes in this repo are run in a python3 virtual environment. As roslaunch interferes with this, the nodes are launched as python scripts

## Installation
Create a python3 venv as an overlay of the ROS Noetic python interpreter (in order to have access to rospy components). From your catkin workspace.
```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
cd src/box_code
mkdir models
mkdir box_venv
python3 -m venv --system-site-packages box_venv
source box_venv/bin/activate
pip3 install -r requirements.txt
# This code was tested with the following installation of pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

```
Put your model files (.pt and .yaml) in the `models` folder.   
Detection messages can be built from [https://github.com/mats-robotics/detection_msgs](https://github.com/mats-robotics/detection_msgs)    
This repo depends on the [yolov5](https://github.com/ultralytics/yolov5.git) submodule, at commit `23701eac7a7b160e478ba4bbef966d0af9348251`
