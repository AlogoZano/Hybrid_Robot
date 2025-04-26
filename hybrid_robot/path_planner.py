import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
import heapq
from scipy.ndimage import grey_dilation

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('a_star_planner')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_cb, 10)
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_cb, 10)

        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        self.occupancy_grid = None

        # Puntos de prueba por ahora (DE MAPA, o sea en metros ps)
        # Ya no son de prueba wuuu
        self.start_world = None
        self.goal_world = None  

    def initial_pose_cb(self, msg):
        self.start_world = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        self.get_logger().info(f"Start pose: {self.start_world}")

        if self.start_world and self.goal_world and self.occupancy_grid: #Muy importante pa empezar a planear
            self.plan_and_publish_path()

    def goal_pose_cb(self, msg):
        self.goal_world = (
            msg.pose.position.x,
            msg.pose.position.y
        )
        self.get_logger().info(f"Goal pose: {self.goal_world}")

        if self.start_world and self.goal_world and self.occupancy_grid: #Muy importante pa empezar a planear
            self.plan_and_publish_path()


    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Recibo Mapa")
        self.occupancy_grid = msg

    def inflate_obstacles(self, grid, inflation_radius_cells):
        structure = np.ones((2 * inflation_radius_cells + 1, 2 * inflation_radius_cells + 1))
        inflated = grey_dilation(grid, footprint=structure)
        
        inflated[grid >= 50] = 100 #Cambiable pa que no choque, aunque el radio jala chido
        return inflated

##################### A* ####################

    def world_to_map(self, x, y, origin, resolution):
        mx = int((x - origin[0]) / resolution) #Normalizar a mapa
        my = int((y - origin[1]) / resolution)
        return mx, my

    def map_to_world(self, mx, my, origin, resolution):
        x = mx * resolution + origin[0] #Regresar a mundo, el origen AFECTA MUCHO
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

        inflated_grid = self.inflate_obstacles(grid, inflation_radius_cells=int(0.2 / resolution))
        path_pixels = self.a_star(start, goal, inflated_grid, width, height)
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
        self.start_world = None
        self.goal_world = None
        self.get_logger().info(f"Path con {len(poses)} poses.")

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
