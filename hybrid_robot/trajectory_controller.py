import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped, Pose2D, PoseStamped
from nav_msgs.msg import Odometry, Path
import tf2_ros
import transforms3d
import numpy as np


class TrajFollower(Node):
    def __init__(self):
        super().__init__('potential_map')

        """Suscriptions"""
        self.laser = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.pos_cb, 10)
        #self.path_sub = self.create_subscription(Path, '/rrt_path', self.path_cb, 10)


        """Publishers"""
        self.robot_speed = self.create_publisher(Twist, 'cmd_vel', 10)

        """Timers """
        self.speed_timer = self.create_timer(0.02, self.speed_timer_cb) 

        """Variables"""
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.theta_rob = 0.0

        self.waypoints = [[0.0, 0.0], [0.5, 0.0], [0.6, -0.3], [-1.5, -0.5]]
        self.prev_waypoint = 0
        self.next_waypoint = 1

        """Messages"""
        self.speed = Twist()

        #Lidar data#
        self.ranges = []
        self.deltaAng = 0.0
        self.angles = []

        #Trajectory error#
        self.error = 0.0
        self.prev_error = 0.0

        #Controller#
        self.k_p = 1.0
        self.k_i = 0.0
        self.k_d = 0.1
        self.integral_error = 0.0 

        self.v_lin = 0.2
        self.v_ang = 0.0
        self.prev_v_ang = 0.0 

        self.ts = 0.02


    def pos_cb(self, msg):
         self.pose_x = msg.pose.pose.position.x
         self.pose_y = msg.pose.pose.position.y

         w = msg.pose.pose.orientation.w
         x = msg.pose.pose.orientation.x
         y = msg.pose.pose.orientation.y
         z = msg.pose.pose.orientation.z

         q = [w, x, y, z]

         _, _, self.theta_rob = transforms3d.euler.quat2euler(q)        

    
    def lidar_callback(self, msg):
        self.Fx_rep = 0.0
        self.Fy_rep = 0.0

        self.ranges = np.asarray(msg.ranges)
        self.ranges[np.isinf(self.ranges)] = 3.5

        self.deltaAng = msg.angle_increment
        self.angles = np.arange(msg.angle_min,msg.angle_max,self.deltaAng)
        

        for i, deg in enumerate (self.angles):
            if (self.ranges[i]<1.0):
                self.Fx_rep += (1/self.ranges[i])**2 * np.cos(deg)
                self.Fy_rep += (1/self.ranges[i])**2 * np.sin(deg)

    def calculate_error(self):
        #Vector of waypoints
        vx = self.waypoints[self.next_waypoint][0] - self.waypoints[self.prev_waypoint][0]
        vy = self.waypoints[self.next_waypoint][1] - self.waypoints[self.prev_waypoint][1]

        #Vector to be projected
        ux = self.pose_x - self.waypoints[self.prev_waypoint][0]
        uy = self.pose_y - self.waypoints[self.prev_waypoint][1]

        #Projection calculation
        v_mode = vx**2 + vy**2

        proj_factor = ((vx*ux)+(vy*uy))/v_mode

        proj_vector_x = proj_factor*vx
        proj_vector_y = proj_factor*vy

        #Error calculation
        error_x = ux - proj_vector_x
        error_y = uy - proj_vector_y

        error = np.linalg.norm([error_x, error_y])

        sign = vx*uy - vy*ux

        return -error*np.sign(sign)
    
    def calculate_distance_next_point(self):
        dx = self.waypoints[self.next_waypoint][0] - self.pose_x
        dy = self.waypoints[self.next_waypoint][1] - self.pose_y
        return np.sqrt(dx**2 + dy**2)


    def speed_timer_cb(self):
        self.error = self.calculate_error()

        print("x: ", self.pose_x, "   y: ", self.pose_y)
        
        # Path direction
        vx = self.waypoints[self.next_waypoint][0] - self.waypoints[self.prev_waypoint][0]
        vy = self.waypoints[self.next_waypoint][1] - self.waypoints[self.prev_waypoint][1]
        desired_theta = np.arctan2(vy, vx)

        # Heading error
        heading_error = desired_theta - self.theta_rob
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize

        # CONTROL (PD)
        derivative = (self.error - self.prev_error) / self.ts
        self.integral_error += self.error * self.ts #Integral gain no usado

        angular_corr = (self.k_p * self.error +
                        self.k_i * self.integral_error +
                        self.k_d * derivative)

        self.speed.angular.z = angular_corr + heading_error

        # Clippeo de v_ang (en prueba)
        self.speed.angular.z = np.clip(self.speed.angular.z, -1.5, 1.5)
        self.speed.linear.x = self.v_lin

        distance = self.calculate_distance_next_point()

        if distance <= 0.15:
            print("Waypoint reached!")
            self.next_waypoint += 1
            self.prev_waypoint += 1

            if self.next_waypoint >= len(self.waypoints):
                self.speed.angular.z = 0.0
                self.speed.linear.x = 0.0
                self.prev_waypoint = len(self.waypoints)-2
                self.next_waypoint = len(self.waypoints)-1

        self.robot_speed.publish(self.speed)
        self.prev_error = self.error


           


def main(args=None):
    rclpy.init(args=args)
    PotMap = TrajFollower()
    rclpy.spin(PotMap)
    PotMap.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()