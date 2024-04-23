import carla
import time
import numpy as np
from numpy.linalg import norm
from splines.ParameterizedCenterline import ParameterizedCenterline
from Logger import Logger
import math

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.centerline = ParameterizedCenterline()
        self.centerline.from_file("/home/ezele2/Desktop/home/Race/waypoints/t1_triple")
        self.progress = None  # Car is spawned in the middle of the track.
        
        self.X = None
        self.Y = None
        self.yaw = None
        self.vx = None
        self.vy = None
        
        self.steps = 0
        self.error = None
        self.last_error = 0
        self.cum_error = 0
        self.cmd_steer = 0
        self.cmd_throttle = 0
        self.logger = Logger("data-mydrive.csv")

    def progress_bound(self):
        """
        Return a bound on the possible progress of the vehicle in order. Such a bound
        is required in order to use the ParameterizedCenterline projection method, but
        it need not be tight.
        """
        if self.progress is None:
            return None
        
        lower = (self.progress-2)  % self.centerline.length
        upper = (self.progress+2) % self.centerline.length

        return (lower, upper) if lower < upper else (upper, lower)

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation. Times out in 10s.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s, in global coordinates
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        self.X = transform.location.x
        self.Y = transform.location.y
        self.yaw = np.deg2rad(transform.rotation.yaw)

        # Velocities in vehicle coordinates (y is lateral).
        # Positive lateral velocity is to the left.
        self.vx = vel.x * np.cos(-self.yaw) - vel.y * np.sin(-self.yaw)
        self.vy = vel.x * np.sin(-self.yaw) + vel.y * np.cos(-self.yaw)

        curr_vel = math.sqrt(self.vx **2 + self.vy **2)

        self.progress, dist = self.centerline.projection(self.X, self.Y, bounds=self.progress_bound())
        self.error = dist * self.centerline.error_sign(self.X, self.Y, self.progress)
        
        print(self.steps)
        print("progress: ", self.progress)
        print("error: ", self.error)

        print("X: ", self.X)
        print("Y: ", self.Y)
        print("Yaw: ", self.yaw)

        print("vx: ", self.vx)
        print('vy: ', self.vy)

        control = carla.VehicleControl()

        ### PD control ###
        D = self.error - self.last_error
        P = self.error
        kD = 15
        kP = 0.5

        control.steer = kD * D + kP * P
        control.throttle = self.longititudal_controller(self.X, self.Y, curr_vel, self.cmd_steer, waypoints)
        #control.throttle = 0.4
        ### end PID control ###

        control.steer = control.steer
        
        print("D: ", D)
        print("P: ", P)
        print("Dt: ", D*kD)
        print("Pt: ", P*kP)
        print("control: ", control.throttle, control.steer)

        self.steps += 1
        self.last_error = self.error
        self.cum_error += self.error
        self.cmd_steer = control.steer
        self.cmd_throttle = control.throttle

        #self.logger.log(self)

        return control


    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        MAX_RELATIVE_YAW = 2
        MAX_VEL = 0.76
        MIN_VEL = 0.4
        curvature_param = 0  # Measure of curvature in [0, 1]; higher means more curve.

        curvature_param = (self.centerline.curvature(self.progress + 5) + self.centerline.curvature(self.progress + 10)+self.centerline.curvature(self.progress + 15) + self.centerline.curvature(self.progress + 20)) * 5

        #curvature_param = self.centerline.curvature(self.progress)
        
        curvature_param = max(0, curvature_param)
        curvature_param = min(1, curvature_param)

        target_velocity = (1-curvature_param)*MAX_VEL + curvature_param*MIN_VEL
        
        print("curvature_param: ", curvature_param)

        if (curvature_param > 0.7):
            return min(target_velocity, 0.39) 
        
        #print("yaw_magnitude: ", yaw_magnitude)
        #print("target_velocity: ", target_velocity)

        return target_velocity

def get_alpha(lk1, lk2, curr_yaw):
    """
    lookahead_point: nparray in ego coordinates
    curr_yaw: yaw of car in rad, float
    """

    avgy = (lk1[1] + lk2[1])/2
    avgx = (lk1[0] + lk2[0])/2
    alpha = np.arctan(avgy / avgx)    
    alpha = normalize_angle(alpha)
    return alpha

def to_ego_frame(point, ego_point, ego_yaw):
    rotation = np.array([
        [np.cos(ego_yaw), np.sin(ego_yaw)],
        [-np.sin(ego_yaw), np.cos(ego_yaw)]
    ])
    translation = np.array([
        [point[0] - ego_point[0]],
        [point[1] - ego_point[1]]
    ])
    ego_frame_point = rotation @ translation
    
    return float(ego_frame_point[0]), float(ego_frame_point[1])

def get_polynomial(inter_points, degree) -> np.poly1d:
    xs = [p[0] for p in inter_points]
    ys = [p[1] for p in inter_points]
    p = np.polyfit(xs, ys, degree)
    
    return np.poly1d(p)

def climb_polynomial(p: np.poly1d, radius, eps=0.1, step=0.01):
    """
    Walk `dist` length along the polynomial `p`. Return the x,y value at that point.
    """
    x = 0
    distance = 0
    while radius - distance > eps:
        distance += dist((x, p(x)), (x+step, p(x+step)))
        x += step
    return x, p(x)

def normalize_angle(angle):
    while angle > np.pi/2:
        angle -=  np.pi
    while angle < -np.pi/2:
        angle +=  np.pi
    return angle

def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 
                + (point1[1] - point2[1])**2)