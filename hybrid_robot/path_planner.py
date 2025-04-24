import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import heapq

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('a_star_planner')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        self.occupancy_grid = None

        # Puntos de prueba por ahora (DE MAPA, o sea en metros ps)
        self.start_world = (0.0, 0.0)  
        self.goal_world = (1.0, 1.0)   

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Received map")
        self.occupancy_grid = msg
        self.plan_and_publish_path()

##################### A* ####################

    def world_to_map(self, x, y, origin, resolution):
        mx = int((x - origin[0]) / resolution) #Normalizar a mapa
        my = int((y - origin[1]) / resolution)
        return mx, my

    def map_to_world(self, mx, my, origin, resolution):
        x = mx * resolution + origin[0]
        y = my * resolution + origin[1]
        return x, y

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Distancia Manhattan
    
    def get_neighbors(self, node, width, height):
        x, y = node
        neighbors = [(x+dx, y+dy) for dx, dy in 
                    [(-1,0), (1,0), (0,-1), (0,1)]]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height]

    def a_star(self, start, goal, grid, width, height):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current, width, height):
                if grid[neighbor[1]][neighbor[0]] >= 50:
                    continue  #Hay obstaculo

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

    def plan_path(self, occupancy_grid: OccupancyGrid, start_world, goal_world):
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        resolution = occupancy_grid.info.resolution
        origin = (
            occupancy_grid.info.origin.position.x,
            occupancy_grid.info.origin.position.y
        )

        grid = np.array(occupancy_grid.data).reshape((height, width))

        start = self.world_to_map(start_world[0], start_world[1], origin, resolution)
        goal = self.world_to_map(goal_world[0], goal_world[1], origin, resolution)

        path_pixels = self.a_star(start, goal, grid, width, height)
        path_world = [self.map_to_world(x, y, origin, resolution) for x, y in path_pixels]

        poses = []
        for x, y in path_world:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0  # Orientacion por defecto mano
            poses.append(pose)

        return poses

    def plan_and_publish_path(self):
        if not self.occupancy_grid:
            return

        poses = self.plan_path(self.occupancy_grid, self.start_world, self.goal_world)

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = poses

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(poses)} poses.")

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
