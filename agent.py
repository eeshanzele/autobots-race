import carla
import time
import numpy as np
import math
from numpy.linalg import norm

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle

        self.X = None
        self.Y = None
        self.velx = None
        self.vely = None
        self.yaw = None

        self.steps = 0
        self.error = None
        self.last_error = 0
        self.cum_error = 0
        self.cmd_steer = 0
        self.cmd_throttle = 0


    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s

        self.X = transform.location.x
        self.Y = transform.location.y
        self.yaw = np.deg2rad(transform.rotation.yaw)
        self.vx = vel.x * np.cos(-self.yaw) - vel.y * np.sin(-self.yaw)
        self.vy = vel.x * np.sin(-self.yaw) + vel.y * np.cos(-self.yaw)

        curr_vel = math.sqrt(self.vx **2 + self.vy ** 2)

        target_vel = longititudal_controller(self.X, self.Y, curr_vel, self.yaw, waypoints)
        target_yaw = pure_pursuit_lateral_controller(self.X, self.Y, curr_vel, self.yaw, waypoints)
        
        control = carla.VehicleControl()

        steering_error = target_yaw - self.yaw

        print("Target Yaw ", target_yaw)
        print("Current Yaw ", self.yaw)
        print("Steering Error ", steering_error)

        if (steering_error > np.pi):
            steering_error -= 2 * np.pi
        elif steering_error < np.pi:
            steering_error += 2 * np.pi
        
        self.error = steering_error
        
        ### PD control ###
        D = self.error - self.last_error
        P = self.error
        kD = 5
        kP = 0.2

        control.steer = kD * D + kP * P

        print("Steering Output", control.steer)
        ### end PID control ###

        acc = 0.01
        if (curr_vel != 0):
            acc = (target_vel - curr_vel)

        if (acc > 0):
            control.throttle = min(acc, 1)
        else:
            control.brake = min(-acc, 1) 

        self.steps += 1
        self.last_error = self.error
        self.cum_error += self.error
        self.cmd_steer = control.steer
        self.cmd_throttle = control.throttle

        return control

def longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

    MAX_RELATIVE_YAW = 0.27  # I recorded on one run that that 0.27 is the max relyaw experienced.
    MAX_VEL = 14
    MIN_VEL = 8
    curvature_param = 0  # Measure of curvature in [0, 1]; higher means more curve.

    next_waypoint = np.array(future_unreached_waypoints[0])
    way_rel = to_ego_frame(next_waypoint, (curr_x, curr_y), curr_yaw)

    yaw_magnitude = abs(get_alpha(way_rel, curr_yaw))
    curvature_param = yaw_magnitude / (MAX_RELATIVE_YAW) 

    curvature_param = max(0, curvature_param)
    curvature_param = min(1, curvature_param)

    target_velocity = (1-curvature_param)*MAX_VEL + curvature_param*MIN_VEL 
    
    #print("yaw_magnitude: ", yaw_magnitude)
    #print("curvature_param: ", curvature_param)
    #print("target_velocity: ", target_velocity)

    return target_velocity


def pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):
    LOOKAHEAD_N = 4
    LOOKAHEAD_DIST = 5
    INTERP_DEG = 3

    curr = np.array([curr_x, curr_y])

    # Find a polynomial in the vehicle coordinate frame.
    interp_points = future_unreached_waypoints[:LOOKAHEAD_N]
    #print(interp_points)
    interp_points_local = [[0., 0.]]  + [to_ego_frame([a, b], curr, curr_yaw) for [a, b, c] in interp_points]
    #print("LOCAL", interp_points_local)

    traj_poly: np.poly1d = get_polynomial(interp_points_local, INTERP_DEG)

    # Calculate required steering angle.
    lookahead_point = climb_polynomial(traj_poly, LOOKAHEAD_DIST)
    alpha = get_alpha(lookahead_point, curr_yaw)

    L = 1.75 ## to change

    relative_yaw = math.atan2(2 * L * np.sin(alpha), norm(lookahead_point))

    return relative_yaw
    
def get_alpha(lookahead_point, curr_yaw):
    """
    lookahead_point: nparray in ego coordinates
    curr_yaw: yaw of car in rad, float
    """
    alpha = np.arctan(lookahead_point[1] / lookahead_point[0])    
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

