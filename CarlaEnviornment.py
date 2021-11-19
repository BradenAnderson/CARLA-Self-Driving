import glob
import os
import sys
import numpy as np
import cv2
import pygame
import random
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(f"Index error!!!")

import carla


class CarEnviornment:

    #front_camera = None

    def __init__(self, show_camera=False, img_width=800, img_height=600, max_seconds_per_episode=30, car_type="Cybertruck", send_extra_info=False):
        self.client = carla.Client("localhost",2000)

        self.show_camera = show_camera
        self.img_width = img_width
        self.img_height = img_height
        self.max_seconds_per_episode = max_seconds_per_episode
        self.car_type = car_type
        self.front_camera = None
        self.trouble_shoot_coutner = 1

        # Extra info that may be desired from the enviornment
        self.velocity_list = []
        self.send_extra_info = send_extra_info

        # Max time a network call is allowed before blocking it and raising a timeout exceed error
        self.client.set_timeout(seconds = 5.0)

        # Get the world object that is currently active in the simulation.
        self.world = self.client.get_world()

        # blueprint_library will be a list of actor blueprints available.
        # this list will make it easy to spawn actors into the simulation world.
        self.blueprint_library = self.world.get_blueprint_library()

        # Grab the first element in the list (which there is likely only one item in the list after filtering).
        self.car_blueprint = self.blueprint_library.filter(f"{self.car_type}")[0]

    def reset(self):

        self.actor_list = []  # List to track actors that we spawn. Needed so we can destory them all at the end.
        self.velocity_list = []
         
        # Vehicle setup
        self.vehicle_transform = random.choice(self.world.get_map().get_spawn_points())                           # Randomly select a spawn location for the vehicle to spawn at.
        self.vehicle = self.world.spawn_actor(blueprint = self.car_blueprint, transform = self.vehicle_transform) # Spawn the car into the world as an actor. 
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))                                 # Set the vehicle to drive forward at full speed.
        self.actor_list.append(self.vehicle)                                                                      # Any time we spawn anything, we need to keep track of it in the actor_list!
                                                                                                                  # (to ensure proper clean up).

        # Camera setup
        self.rgb_camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')    # Get the blueprint for the rgb camera
        self.rgb_camera_blueprint.set_attribute("image_size_x", f"{self.img_width}")    # Set the cameras image width.
        self.rgb_camera_blueprint.set_attribute("image_size_y", f"{self.img_height}")   # Set the cameras image height.
        self.rgb_camera_blueprint.set_attribute("fov", "110")                           # Set the cameras field of view
        self.camera_transform = carla.Transform(carla.Location(x = 2.75, z = 2.5))      # Get the camera spawn point, relative to the center of the vehicle. 
        self.camera = self.world.spawn_actor(blueprint = self.rgb_camera_blueprint,     # Spawn the camera.
        transform = self.camera_transform, attach_to= self.vehicle)
        self.actor_list.append(self.camera)                                             # Add camera to actor list, for proper cleanup later.
        self.camera.listen(lambda image_data : self.process_image(image_data))          # Listen to the images captured by the camera.
    
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # Send a command to help CARLA intialize faster. (Some people say that applying some control
                                                                                  # right away will make CARLA start responding to commands faster?)
 
        # Four second sleep to make sure CARLA is initalized before we start doing anything
        time.sleep(4)
        
        # Collision sensor setup
        self.collision_history = []                                                                  # List to hold any collision events
        self.collision_sensor_blueprint = self.blueprint_library.find("sensor.other.collision")      # Get the blueprint for the collision sensor
        self.collision_sensor_transform = carla.Transform(carla.Location())                          # Get the location to place the collision sensor
        self.collision_sensor = self.world.spawn_actor(blueprint = self.collision_sensor_blueprint,  # Spawn the collision sensor (at the center of the vehicle).
        transform = self.collision_sensor_transform, attach_to = self.vehicle) 
        self.actor_list.append(self.collision_sensor)                                                # Add the collision sensor to the actor list!
        self.collision_sensor.listen(lambda collision_event : self.collision_data(collision_event))  # Listen to the data that is generated by the collision sensor.
        
        # Waiting loop to make sure camera is initialized before moving on.
        # self.front_camera will stop being none the first time
        # self.process_image is called in the lambda for self.camera.listen
        while self.front_camera is None:
            time.sleep(0.01)

        # Store episode starting time.
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # Send a command to help CARLA intialize faster. (Some people say that applying some control
                                                                                  # right away will make CARLA start responding to commands faster?)
        return self.front_camera
    
    # This function updates the collision sensor list whenever a collision occurs.
    def collision_data(self, event):
        self.collision_history.append(event)

    # This function takes in the images sent from the cars camera and stores it 
    # in the self.front_camera attribute.
    def process_image(self, image):

        # image.raw_data gives an array of 32-bit RBGA pixels
        # The CARLA image.raw_data starts off as a flattened array
        img_array = np.array(image.raw_data)                               

        image_reshaped = np.reshape(img_array, (image.height, image.width, 4)) # Depth is four because this is an RGBA camera (not just RGB).                                                  
        image_3D = image_reshaped[:, :, :3]                                    # Cutting out the fourth "alpha" dimension, so we are just left with RGB.

        # If we want to display the view from the cars camera. 
        if self.show_camera:
            cv2.imshow("car camera", image_3D) # Show the image returned by the cars camera.
            cv2.waitKey(1)                     # Display image for at least 1ms and then close.

        self.front_camera = image_3D

    def step(self, action):

        self.trouble_shoot_coutner +=1
        
        # Applying the new command
        throttle_cmd, steer_cmd = action                                                # Unpack three continuous action commands

        if throttle_cmd >=0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = throttle_cmd, steer=steer_cmd, brake=0.0))
        elif throttle_cmd < 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer=steer_cmd, brake=throttle_cmd))

        velocity = self.vehicle.get_velocity()
        kilometers_per_hour = 3.6 * ((velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)

        #if self.trouble_shoot_coutner % 50 == 0:
        #    print(f"\n=============== counter: {self.trouble_shoot_coutner}==========================")
        #    print(f"Velocity x: {velocity.x}")
        #    print(f"Velocity y: {velocity.y}")
        #    print(f"Velocity z: {velocity.z}")
        #    print(f"Kph :{kilometers_per_hour}")

        if self.send_extra_info:
            self.velocity_list.append(kilometers_per_hour)
        

        # REWARD CALCULATION
        # If we have registered any collisions
        if len(self.collision_history) != 0:
            done = True
            reward = -0.5

        # Else calculate reward based on speed
        else:
            if kilometers_per_hour < 48:
                reward = -.005 + 0.000166667*kilometers_per_hour
            elif kilometers_per_hour > 48 and kilometers_per_hour <= 100:
                reward = 0.00115385 + 0.0000384615*kilometers_per_hour
            elif kilometers_per_hour > 100:
                reward = 0.0155263 - 0.000105263*kilometers_per_hour
            
            done = False

        # If more than max_seconds_per_episode seconds have elapsed since the episode started.
        if self.episode_start + self.max_seconds_per_episode < time.time():
            done = True

        if self.send_extra_info and done:
            extra_info = (np.mean(self.velocity_list), np.max(self.velocity_list), len(self.collision_history))
        else:
            extra_info = None

        # next_state, reward, done, extra_info
        return self.front_camera, reward, done, extra_info
