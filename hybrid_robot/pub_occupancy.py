import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import yaml
import numpy as np
from PIL import Image

class ManualMapPublisher(Node):
    def __init__(self):
        super().__init__('manual_map_publisher')

        # Load YAML metadata
        with open('/home/alogo/TE3003B_ws/src/hybrid_robot/maps/test3.yaml', 'r') as f:
            map_metadata = yaml.safe_load(f)

        # Load PGM
        image = Image.open('/home/alogo/TE3003B_ws/src/hybrid_robot/maps/' + map_metadata['image'])
        image = image.convert('L')  # ensure grayscale
        map_array = np.array(image)

        # Convert image to occupancy data
        occupied_thresh = map_metadata['occupied_thresh'] * 255
        free_thresh = map_metadata['free_thresh'] * 255

        occupancy_data = []
        for pixel in map_array.flatten(order='C'):  # Row-major
            if pixel < free_thresh:
                occupancy_data.append(0)
            elif pixel > occupied_thresh:
                occupancy_data.append(100)
            else:
                occupancy_data.append(-1)

        # Build OccupancyGrid
        self.map_msg = OccupancyGrid()
        self.map_msg.header = Header()
        self.map_msg.header.frame_id = 'map'
        self.map_msg.info.resolution = map_metadata['resolution']
        self.map_msg.info.width = image.width
        self.map_msg.info.height = image.height
        self.map_msg.info.origin.position.x = map_metadata['origin'][0]
        self.map_msg.info.origin.position.y = map_metadata['origin'][1]
        self.map_msg.info.origin.position.z = 0.0
        self.map_msg.info.origin.orientation.w = 1.0
        self.map_msg.data = occupancy_data

        self.publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.timer = self.create_timer(1.0, self.publish_map)

    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ManualMapPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
