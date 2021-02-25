# Solution for ACM 216 HW3 Problem 4

import random as rand
from matplotlib import pyplot
import matplotlib
import numpy as np
import math
import scipy.misc
from PIL import Image


# Helper function to normalize a grayscale image to a [-1, 1] range
def normalize_image(img_arr):
    x_res = img_arr.shape[0]
    y_res = img_arr.shape[1]
    norm_img = []
    for x in range(x_res):
        row = []
        for y in range(y_res):
            # Map intensity from [0, 255] -> [-1, 1]
            i = img_arr[x][y][0]
            m = i * 2 / 255 + -1
            # Dark pixels are in the foreground for the Cow image, so flip the
            # negativity of the image so that high intensity pixels are in the
            # foreground (generally speaking)
            m *= -1
            row.append(m)
        norm_img.append(row)
    return np.array(norm_img)


# Helper function to create the initial segmentation state for an image
def create_initial_state(img):
    x_res = img.shape[0]
    y_res = img.shape[1]
    state = []
    for x in range(x_res):
        row = []
        for y in range(y_res):
            row.append(1 if img[x][y] > 0 else -1)
        state.append(row)
    return np.array(state)

# Get the change in Ising Energy caused by flipping the state of the pixel at
# (x_flip, y_flip)
def get_delta_energy(img, state, x_flip, y_flip, theta):
    delta_energy = 0.0
    
    # Change in energy from external 'field' sum. Because every other pixel
    # remains the same, this is the change of energy at (x_flip, y_flip).
    # This is -(theta * b_xy * eps_xy) - (-theta * b_xy * eps'_xy), but because
    # eps'_xy = -eps_xy, this is just -2(theta * b_xy * eps_xy)
    delta_energy += -2 * theta * img[x_flip, y_flip] * state[x_flip, y_flip]
    
    # Change from neighboring pixels.
    # Only the pixel at (x_flip, y_flip) changes, so we need only consider the
    # change in energy from interactions with neighbors of that pixel.
    neighbor_delta = 0.0
    x_res = img.shape[0]
    y_res = img.shape[1]    
    s = state[x_flip][y_flip]
    if x_flip < x_res - 1: # Add neighbor to the left
        xn = x_flip + 1
        yn = y_flip
        sn = state[xn][yn]
        neighbor_delta -= s * sn    
    if x_flip > 0: # Add neighbor to the right
        xn = x_flip - 1
        yn = y_flip
        sn = state[xn][yn] 
        neighbor_delta -= s * sn    
    if y_flip < y_res - 1: # Add neighbor above
        xn = x_flip
        yn = y_flip + 1
        sn = state[xn][yn]
        neighbor_delta -= s * sn    
    if y_flip > 0: # Add neighbor below
        xn = x_flip
        yn = y_flip - 1
        sn = state[xn][yn]  
        neighbor_delta -= s * sn
    
    delta_energy += neighbor_delta
        
    return delta_energy

# Get the change in Ising Energy caused by flipping the state of the pixel at
# (x_flip, y_flip). Whereas get_delta_energy returns the energy as defined by
# the problem set, this function considers all neighbors within 'radius' pixels
# of the pixel to flip (i.e. considers further neighbors) including diagonals
def get_delta_energy_radius(img, state, x_flip, y_flip, theta, radius):
    delta_energy = 0.0
    
    # Change in energy from external 'field' sum. Because every other pixel
    # remains the same, this is the change of energy at (x_flip, y_flip).
    # This is -(theta * b_xy * eps_xy) - (-theta * b_xy * eps'_xy), but because
    # eps'_xy = -eps_xy, this is just -2(theta * b_xy * eps_xy)
    delta_energy += -2 * theta * img[x_flip, y_flip] * state[x_flip, y_flip]
    
    # Change from neighboring pixels.
    # Only the pixel at (x_flip, y_flip) changes, so we need only consider the
    # change in energy from interactions with neighbors of that pixel.
    neighbor_delta = 0.0
    x_res = img.shape[0]
    y_res = img.shape[1]    
    s = state[x_flip][y_flip]
    for dx in range(1, radius + 1):
        for dy in range(1, radius + 1):
            if x_flip - dx >= 0: 
                xn = x_flip - dx
                yn = y_flip
                sn = state[xn][yn]
                neighbor_delta -= s * sn    
            if x_flip + dx < x_res:
                xn = x_flip + dx
                yn = y_flip
                sn = state[xn][yn] 
                neighbor_delta -= s * sn    
            if y_flip - dy >= 0: 
                xn = x_flip
                yn = y_flip - dy
                sn = state[xn][yn]
                neighbor_delta -= s * sn    
            if y_flip + dy < y_res:
                xn = x_flip
                yn = y_flip + dy
                sn = state[xn][yn]  
                neighbor_delta -= s * sn
    
    delta_energy += neighbor_delta
        
    return delta_energy

# Get h(e^(delta energy)), which is the probability that we take a transition
# that causes a change in energy equal to delta energy. 'state' is the current
# classifications of the image, img is the original image, and (x_flip, y_flip)
# is the pixel we are considering flipping.
def get_h_value(img, state, x_flip, y_flip, theta, T):
    deltaEnergy = get_delta_energy(img, state, x_flip, y_flip, theta, 1)
    try:
        u = math.exp(deltaEnergy / T)
    except OverflowError as err:
        return 1
    # Using h(u) = u / (u + 1)
    return u / (u + 1)


# Runs the metropolis algorithm with initial segmentation state init_state
# and parameters theta and T on an image img for n_steps steps.
def run_metropolis(init_state, img, theta, T, n_steps, verbose, log_file):
    x_res = img.shape[0]
    y_res = img.shape[1]    
    # Step 0: initialized state given
    state = np.copy(init_state)
    num_steps_taken = 0
    for s in range(n_steps): # Step n -> n + 1
        if verbose:
            print("On step", s + 1, "out of", n_steps)
        # Neighbors of X_n are states that differ in their classification at
        # exactly one pixel, selected uniformly. With this, the number of
        # neighbors at each state are equal, so their ratios are equal also.
        x = rand.randint(0, x_res - 1)
        y = rand.randint(0, y_res - 1)
        U = rand.random()
        # Get h, the probability of taking the transition caused by flipping
        h = get_h_value(img, state, x, y, theta, T)
        if U < h:
            # Take the transition
            state[x][y] *= -1
            num_steps_taken += 1
            
    # Log what percent of transitions were taken with the parameters
    pct_transition = num_steps_taken / n_steps * 100
    log_file.write("Percent of transitions accepted: " 
                   + str(pct_transition) + "%\n\n")
    return state


# Load/normalize image and set up initial state
img_name = "Cow"
out_path = "Output/"
raw_img = pyplot.imread(img_name + ".jpg")
norm_img = normalize_image(raw_img)
init_state = create_initial_state(norm_img)
matplotlib.image.imsave(out_path + img_name + "_Initial_Segmentation.jpg", 
                        init_state, cmap='gray',vmin=-1, vmax=1)

# Problem parameters
n_steps = 100000
theta_vals = [0.001, 0.1, 1, 10, 25, 100]
T_vals = [0.001, 0.1, 1, 10, 25, 100]
# Run the metropolis algorithm to generate segmentations for each set of params
log_file = open(out_path + "CowLog.txt", 'w')
for theta in theta_vals:
    for T in T_vals:
        print("On theta =", theta, " and T =", T)
        log_file.write("On theta = " + str(theta) + " and T = " + str(T) + "\n")
        final_state = run_metropolis(init_state, norm_img, theta, T, n_steps,
                                     False, log_file)
        im_name = out_path + img_name + "Theta=" + str(theta) + "_T=" + str(T) + ".jpg"
        matplotlib.image.imsave(im_name, final_state, cmap='gray',
                                vmin=-1, vmax=1)
        log_file.close()
        log_file = open(out_path + "CowLog.txt", 'a')
log_file.close()