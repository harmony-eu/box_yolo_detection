## Usage
The nodes in this repo are run in a python3 virtual environment. As roslaunch interferes with this, the nodes are launched as python scripts

## Installation
- Create a Python3 `virtualenv` as an overlay of the ROS Noetic Python interpreter (in order to have access to `rospy` components). In the following it assumed that the path to the `virtualenv` folder is `$BOX_YOLO_ENV`:
    ```bash
    export BOX_YOLO_ENV=~/.virtualenvs/box_yolo;
    mkdir -p ${BOX_YOLO_ENV};
    python3 -m venv --system-site-packages ${BOX_YOLO_ENV};
    source ${BOX_YOLO_ENV}/bin/activate;
    ```
- In the following, it is assumed that you have created a ROS Noetic `catkin` workspace called `$BOX_CATKIN_WS`, in which this package should be installed.
    ```bash
    source /opt/ros/noetic/setup.bash
    source ${BOX_CATKIN_WS}/devel/setup.bash
    cd ${BOX_CATKIN_WS}/src;
    # Clone this repo.
    git clone git@github.com:harmony-eu/box_yolo_detection.git;
    cd ${BOX_CATKIN_WS}/src/box_yolo_detection;
    # Clones the submodule of this repo (the original `yolov5` repo).
    git submodule update -i -r;
    pip3 install -r requirements.txt
    # Installs the required PyTorch dependencies (tested with the versions below).
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
    ```
- Put your model files (`.pt` and `.yaml`) in the `models` folder.
- To run the nodes in this repo, you will need to have built ROS messages of type `detection_msgs`. To do so, clone the [detection_msgs](https://github.com/mats-robotics/detection_msgs) repo in your `catkin` workspace and build it:
    ```bash
    cd ${BOX_CATKIN_WS}/src;
    git clone https://github.com/mats-robotics/detection_msgs.git;
    catkin build
    ```
