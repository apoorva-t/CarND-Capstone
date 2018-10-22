from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy
import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_SPEED = 20  # in MPH

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
    			 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        #kp = 3
        kp = 0.3
        #ki = 0.5
        ki = 0.1
        #kd = 0.5 
        kd = 0.
        mn = 0.0
        mx = 0.2
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
    	self.wheel_radius = wheel_radius

    	tau = 0.5  # 1/(2pi * tau)
    	ts = 0.02  # sample time
    	self.vel_lpf = LowPassFilter(tau, ts) # to filter out high frequency noise in velocity being passed

    	self.steer_lpf = LowPassFilter(tau=0.5, ts=0.02)

    	self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # Return throttle, brake, steer
        if not dbw_enabled:
        	#rospy.loginfo('dbw disabled')
        	self.throttle_controller.reset()
        	return 0.0,0.0,0.0

        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        steering = self.steer_lpf.filt(steering)

        vel_error = min(linear_vel, MAX_SPEED*ONE_MPH) - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        rospy.loginfo('Value of throttle from pid: %f, vel_err: %f', throttle, vel_error)
        
        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
        	throttle = 0
        	brake = 700 # N*m - to hold the car in place if we are stopped at a light
        	rospy.loginfo('Values of throttle= 0, brake = 400, steering = %d', steering)
        elif throttle < .1 and vel_error < 0.0:
        	throttle = 0.0
        	decel = max(self.decel_limit, vel_error)
        	brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        	rospy.loginfo('Values of throttle: 0, brake: %d, steering: %d', brake, steering)

        return throttle, brake, steering

        '''
        if vel_error < 0.0:
        	brake = 20 * abs(vel_error)

        if linear_vel < 0.1:
        	throttle = 0.0
        	brake = 400 # N-m
        return throttle, brake, steering
        '''
