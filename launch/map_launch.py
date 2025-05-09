from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': '/home/alogo/TE3003B_ws/src/hybrid_robot/maps/test3.yaml'
            }]
        )
        
    ])
