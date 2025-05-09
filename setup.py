from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hybrid_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch','*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alogo',
    maintainer_email='alogo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_controller = hybrid_robot.trajectory_controller:main',
            'pub_occupancy = hybrid_robot.pub_occupancy:main',
            'path_planner = hybrid_robot.path_planner:main'

        ],
    },
)
