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
        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_cb, 10)


        """Publishers"""
        self.robot_speed = self.create_publisher(Twist, 'cmd_vel', 10)

        """Timers """
        self.speed_timer = self.create_timer(0.02, self.speed_timer_cb) 
        self.speed_timer.cancel()

        """Variables"""
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.theta_rob = 0.0

        #self.waypoints_test = [[-0.12999991655349774, -0.03999992474913583], [-0.12999991655349774, -0.08999992549419389], [-0.12999991655349774, -0.13999992623925195], [-0.12999991655349774, -0.18999992698431], [-0.12999991655349774, -0.23999992772936807], [-0.12999991655349774, -0.2899999284744261], [-0.12999991655349774, -0.3399999292194842], [-0.12999991655349774, -0.38999992996454225], [-0.12999991655349774, -0.4399999307096003], [-0.12999991655349774, -0.48999993145465837], [-0.12999991655349774, -0.5399999321997164], [-0.12999991655349774, -0.5899999329447745], [-0.12999991655349774, -0.6399999336898325], [-0.12999991655349774, -0.6899999344348906], [-0.12999991655349774, -0.7399999351799487], [-0.12999991655349774, -0.7899999359250067], [-0.12999991655349774, -0.8399999366700648], [-0.12999991655349774, -0.8899999374151228], [-0.12999991655349774, -0.9399999381601809], [-0.12999991655349774, -0.989999938905239], [-0.12999991655349774, -1.039999939650297], [-0.07999991580843968, -1.039999939650297], [-0.02999991506338162, -1.039999939650297], [0.02000008568167644, -1.039999939650297], [0.0700000864267345, -1.039999939650297], [0.12000008717179256, -1.039999939650297], [0.17000008791685062, -1.039999939650297], [0.22000008866190868, -1.039999939650297], [0.27000008940696674, -1.039999939650297], [0.3200000901520248, -1.039999939650297], [0.37000009089708286, -1.039999939650297], [0.4200000916421409, -1.039999939650297], [0.470000092387199, -1.039999939650297], [0.520000093132257, -1.039999939650297], [0.5700000938773151, -1.039999939650297], [0.6200000946223732, -1.039999939650297], [0.6700000953674312, -1.039999939650297], [0.7200000961124893, -1.039999939650297], [0.7700000968575473, -1.039999939650297], [0.8200000976026054, -1.039999939650297], [0.8700000983476635, -1.039999939650297], [0.9200000990927215, -1.039999939650297]]
        self.waypoints = []
        print(self.waypoints)

        self.prev_waypoint = 0
        self.next_waypoint = 1

        """Messages"""
        self.speed = Twist()
        self.path = Path()

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

    def path_cb(self, msg):
        self.path.poses = msg.poses

        if len(self.path.poses):
            print("Received path!!")

            for point in self.path.poses:
                self.waypoints.append([point.pose.position.x, point.pose.position.y])
                
                self.speed_timer = self.create_timer(0.02, self.speed_timer_cb) 

            print("ORIGINALES: ", self.waypoints)
            self.waypoints = self.reduce_waypoints(self.waypoints, 30)
            print("BONITOS: ", self.waypoints)
            print("Of len: ", len(self.waypoints))
                
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

    #Tremenda funcion, escoge puntos solo en esquinas (aparte del de inicio y fin)
    def reduce_waypoints(self, waypoints, angle_threshold_deg=20):
        if len(waypoints) < 3:
            return waypoints

        def angle_between(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return np.arctan2(dy, dx)

        reduced = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            angle1 = angle_between(waypoints[i - 1], waypoints[i])
            angle2 = angle_between(waypoints[i], waypoints[i + 1])
            angle_diff = np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1))   #Angulo entre waypoints
            angle_deg = np.degrees(np.abs(angle_diff))

            if angle_deg >= angle_threshold_deg: #Importante el threshold
                reduced.append(waypoints[i])

        reduced.append(waypoints[-1])
        return reduced


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
        self.speed.angular.z = np.clip(self.speed.angular.z, -3.0, 3.0)
        self.speed.linear.x = self.v_lin

        #No lineal si esta bien chueco
        if abs(heading_error) > np.pi / 4:  
            self.speed.linear.x = 0.0
        else:
            self.speed.linear.x = self.v_lin

        distance = self.calculate_distance_next_point()

        #Aun no me convence esta distancia pero jala bien
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