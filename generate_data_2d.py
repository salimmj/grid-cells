import os
import tensorflow as tf
from scipy import stats
import numpy as np
import math 
from sklearn import preprocessing

# Duration of simulated trajectories (seconds)
T = 15

# Width and height of environment, or diameter for circular environment (meters)
L = 220

# Perimeter region distance to walls (meters)
d = 3

# Forward velocity Rayleigh distribution scale (m/sec)
forward_v_sigma = 13.02

# Rotation velocity Guassian distribution mean (deg/sec)
mu = -0.03 

# Rotation velocity Guassian distribution standard deviation (deg/sec)
angular_v_sigma = 330.12

# Velocity reduction factor when located in the perimeter 
v_reduction_factor = 0.25

# Change in angle when located in the perimeter (deg)
angle_delta = 90

# Simulation-step time increment (seconds)
dt = 0.02

# Number of place cells
N = 256

# Place cell standard deviation parameter (meters)
pc_std = 0.01

# Number of target head direction cells
M = 12

# Head direction concentration parameter 
K = 20

# Gradient clipping threshold
g_c = 10**-5

# Number of trajectories used in the calculation of a stochastic gradient
minibatch_size = 10

# Number of time steps in the trajectories used for the supervised learning task
trajectory_length = 100

# Step size multiplier in the RMSProp algorithm
learning_rate = 10**-5

# Momentum parameter of the RMSProp algorithm
momentum = 0.9

# Regularisation parameter for linear layer
L2_reg = 10**-5

# Total number of gradient descent steps taken
parameter_updates = 300000

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def angle(v1, v2):
    """The acute angle between two vectors"""
    angle = np.arctan2(v2[2], v2[0]) - np.arctan2(v1[2], v1[0])
    if angle > np.pi:
      angle -= 2*np.pi
    elif angle <= -np.pi:
      angle += 2*np.pi
    return angle

def min_dist_angle(position, direction):
    """Distance to the closest wall and its corresponding angle

    Keyword arguments:
    position -- the position (3-dimensional vector)
    direction -- head direction (3-dimensional vector)
    """
    
    # Perpendicular distance to line 
    # Southern Wall z = 0
    s_dist = position[2]
    # Northern Wall z = L
    n_dist = L - position[2]
    # Western Wall x = 0
    w_dist = position[0]
    # Eastern Wall x = L
    e_dist = L - position[0]
    
    wall_dists = [s_dist, n_dist, w_dist, e_dist]
    
    min_pos = np.argmin(wall_dists)
    
    dWall = wall_dists[min_pos]
    
    west_wall = [-1, 0, 0]

    north_wall = [0, 0, 1]

    east_wall = [1, 0, 0]

    south_wall = [0, 0, -1]
    
    walls = [south_wall, north_wall, west_wall, east_wall]
    aWall = angle(direction, walls[min_pos])
    return [dWall, aWall]

def normalize(vec):
    return vec / np.linalg.norm(vec)

def generate_rat_trajectory_for_dataset(steps, sample):
    """Generate a pseudo-random rat trajectory within a L-size square cage

    steps - number of steps for the rat to take
    
    return ->
      position - (samples,3)-shaped matrix holding the 3-dim positions overtime
      velocity - (samples,3)-shaped matrix holding the 3-dim velocities overtime
    """

    # record every steps/sample steps
    frequency = steps/sample
    
    # Initialize parameters for velocity and camera
    v = 20
    dirr = normalize(np.random.rand(3))
    up = np.array([0, 1, 0])
    dt = 0.02
    norm_vec = np.array([0,0,0])

    # create random velocity samples
    random_turn = np.radians(np.random.normal(mu, angular_v_sigma, steps))
    # print(random_turn)
    random_velocity = np.random.rayleigh(forward_v_sigma, steps)
    
    # allocate memory for x, y, and z-components of position and velocity
    hd = np.zeros(steps+1)
    hd[0] = angle(norm_vec, dirr)

    ego_vel = np.zeros((steps, 3))

    position_matrix = np.zeros((steps+1, 3))
    position_matrix[0] = L*np.random.rand(3) # initialize
    velocity_matrix = np.zeros((steps+1, 3))
    
    for step in range(1, steps+1):
        # computes the min distance and corresponding angle for a position
        [dWall, aWall] = min_dist_angle(position_matrix[step-1], dirr)

        # update speed and turn angle 
        if dWall<2 and np.absolute(aWall)<np.pi/2:
            # print('oups')
            angle_vel = aWall/np.absolute(aWall)*(np.pi-np.absolute(aWall)) + random_turn[step-1]
            v = v-0.5*(v-v_reduction_factor) # slow down
        else:
            v = random_velocity[step-1]
            angle_vel = random_turn[step-1]

        ego_vel[step-1][0] = v/100.0
        ego_vel[step-1][1] = angle_vel
        ego_vel[step-1][2] = angle_vel

        low = np.array([0,0,0])
        high = np.array([L,L,L])
        # move.
        position_matrix[step] = position_matrix[step-1] + dirr*v*dt #np.minimum(np.maximum(position_matrix[step-1] + dirr*v*dt, low), high)
        velocity_matrix[step] = dirr*v*dt
        
        # turn the 3D direction vector around y-axis
        dirr = y_rotation(dirr, angle_vel*dt)
        hd[step] = angle(norm_vec, dirr)

    index = [True if i%frequency==0 else False for i in range(steps)]
    position_matrix = np.delete(position_matrix, 1, 1)/100.0 - 1.1

    def bin_mean(arr):
      return stats.binned_statistic(np.arange(steps), arr, 'mean', bins=sample).statistic

    # ego_vel = np.apply_along_axis(bin_mean, 0, ego_vel)

    ego_vel[:,1] = np.sin(ego_vel[:,1])
    ego_vel[:,2] = np.cos(ego_vel[:,2])

    velocity_matrix = velocity_matrix/100.0
    # return init_pos, init_hd, ego_vel, target_pos, target_hd
    return [position_matrix[0], 
            np.array([hd[0]]), 
            ego_vel[index, :], 
            position_matrix[1:][index, :], 
            hd[1:][index]]

def filename_generator(root):
  """Generates lists of files for a given dataset version."""
  basepath = 'square_room_100steps_2.2m_1000000'
  base = os.path.join(root, basepath)
  num_files = 100
  template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
  return [
    os.path.join(base, template.format(i, num_files - 1))
    for i in range(num_files)
  ]

filenames = filename_generator('./my_datasets')

def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

records_per_file = 10000

for filename in filenames:
    tfrecord_writer = tf.io.TFRecordWriter(filename)

    data = [generate_rat_trajectory_for_dataset(800, 100) for _ in range(records_per_file)]

    for index in range(records_per_file):
      # 1. Convert your data into tf.train.Feature
      feature = {
        'init_pos': _float32_feature(data[index][0]),
        'init_hd': _float32_feature(data[index][1]),
        'ego_vel': _float32_feature([val for row in data[index][2] for val in row]), # flatten 
        'target_pos': _float32_feature([val for row in data[index][3] for val in row]), # flatten
        'target_hd': _float32_feature(data[index][4]) # flatten
      }
      # 2. Create a tf.train.Features
      features = tf.train.Features(feature=feature)
      # 3. Createan example protocol
      example = tf.train.Example(features=features)
      # 4. Serialize the Example to string
      example_to_string = example.SerializeToString()
      # 5. Write to TFRecord
      tfrecord_writer.write(example_to_string)