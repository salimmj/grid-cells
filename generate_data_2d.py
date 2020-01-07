import os
import tensorflow as tf
from scipy import stats
import numpy as np
import math 
from sklearn import preprocessing
import concurrent.futures
import logging
import threading
from multiprocessing import Process

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

def angle_3d(v1, v2):
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
    aWall = angle_3d(direction, walls[min_pos])
    return [dWall, aWall]

def normalize(vec):
    return vec / np.linalg.norm(vec)

def rotation(vector,theta):
    """Rotates 2-D vector around y-axis"""
    return np.array([np.cos(theta)*vector[0]-np.sin(theta)*vector[1],np.sin(theta)*vector[0]+np.cos(theta)*vector[1]])

def angle(v1, v2):
    """The acute angle between two vectors"""
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    # angle = angle%(2*np.pi)
    if angle > np.pi:
      angle -= 2*np.pi
    elif angle <= -np.pi:
      angle += 2*np.pi
    return angle

def velocity(vec):
  return np.linalg.norm(vec)/0.16

def head_dir(vec):
  return angle(np.array([1,0]), vec).astype('float32')

def sample(position_matrix):
  assert len(position_matrix) == 816
  position_matrix = position_matrix[[True if i%8==0 else False for i in range(816)]]
  start_pos = position_matrix[0]
  diffs = np.array([position_matrix[i+1] - position_matrix[i] for i in range(101)])
  trans_vels = np.apply_along_axis(velocity, 1, diffs)
  hds = np.apply_along_axis(head_dir, 1, diffs)
  ang_vels = np.array([hds[i+1] - hds[i] for i in range(100)])/0.16
  # print(min(hds), max(hds))
  # print(len(trans_vels), len(hds), len(ang_vels), len(position_matrix[1:-1]))
  ego_vel = np.zeros((100, 3))
  ego_vel[:,0] = trans_vels[:-1]
  ego_vel[:,1] = np.cos(ang_vels)
  ego_vel[:,2] = np.cos(ang_vels)
  return [position_matrix[0], 
          np.array([hds[0]]), 
          ego_vel, 
          position_matrix[1:-1], 
          hds[1:]]


def generate_rat_trajectory(steps):
    """Generate a pseudo-random rat trajectory within a L-size square cage

    steps - number of steps for the rat to take
    
    return ->
      position - (samples,3)-shaped matrix holding the 3-dim positions overtime
      velocity - (samples,3)-shaped matrix holding the 3-dim velocities overtime
    """
    
    # Initialize parameters for velocity and camera
    v = 20
    dirr = normalize(np.random.rand(3))
    up = np.array([0, 1, 0])
    dt = 0.02
    norm_vec = np.array([1,0,0])

    # create random velocity samples
    random_turn = np.radians(np.random.normal(mu, angular_v_sigma, steps))
    # print(random_turn)
    random_velocity = np.random.rayleigh(forward_v_sigma, steps)
    
    hd = np.zeros(steps+1)
    hd[0] = angle_3d(norm_vec, dirr).astype('float32')

    # allocate memory for x, y, and z-components of position and velocity
    position_matrix = np.zeros((steps, 3))
    position_matrix[0] = L*np.random.rand(3) # initialize
    velocity_matrix = np.zeros((steps, 3))
    
    for step in range(1, steps):
        # computes the min distance and corresponding angle for a position
        [dWall, aWall] = min_dist_angle(position_matrix[step-1], dirr)

        # update speed and turn angle 
        if dWall<3 and np.absolute(aWall)<np.pi/2:
            # print('oups')
            angl = aWall/np.absolute(aWall)*(np.pi-np.absolute(aWall)) + random_turn[step]
            v = v-0.25*(v) # slow down
        else:
            v = random_velocity[step]
            angl = random_turn[step]
        
        low = np.array([0,0,0])
        high = np.array([L,L,L])
        # move.
        position_matrix[step] = (position_matrix[step-1] + dirr*v*dt) #np.minimum(np.maximum(position_matrix[step-1] + dirr*v*dt, low), high)
        velocity_matrix[step] = (dirr*v*dt)
        
        # turn the 3D direction vector around y-axis
        dirr = y_rotation(dirr, angl*dt)
        hd[step] = angle_3d(norm_vec, dirr)
        
    # return init_pos, init_hd, ego_vel, target_pos, target_hd
    return (np.delete(position_matrix,1,1)/100.0 - 1.1).astype('float32')

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

def write_record(filename):
    print('writing record', filename)
    tfrecord_writer = tf.io.TFRecordWriter(filename)

    data = [sample(generate_rat_trajectory(816)) for _ in range(records_per_file)]

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
        
def write_records(filenames):
    for filename in filenames:
        write_record(filename)

if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO,
#                         datefmt="%H:%M:%S")

#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         executor.map(write_record, filenames)
#     for filename in filenames:
#         write_record(filename)
    coord = tf.train.Coordinator()
    processes = []
    args = np.array_split(filenames, 4)
    for thread_index in range(4):
        p = Process(target=write_records, args=[args[thread_index]])
        p.start()
        processes.append(p)
    coord.join(processes)