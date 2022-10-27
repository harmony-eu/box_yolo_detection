import os

from glob import glob
from setuptools import setup

package_name = 'box_yolo_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Giulio Schiavi',
    maintainer_email='harmony_asl@todo.todo',
    description='Yolo v5 ROS2 node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'box_yolo_detection = box_yolo_detection.nodes.detect:main'
        ],
    },
)
