# Solution for ACM 216 HW3 Problem 5

import random as rand
from matplotlib import pyplot
import matplotlib
import numpy as np
import math
import pandas as pd 

# Number of counties in the CAtowns.xsl dataset
num_counties = 58

# Artificial penalty imposed on crossing into a different county.
# We cannot avoid going to each county so we must incur this penalty once for
# each county, but this greatly reduces the probability that the
# algorithm selects a path that crosses boarders more than needed.
# For context, this penalty is orders of magnitude larger then the entire
# initial tour cost without boarder penalties.
county_boarder_penalty = 1e9

# Class to store data about a town
class Town:
    def __init__(self, x, y, county):
        self.x = x
        self.y = y
        self.county = county
    
    def __str__(self):
        return f'Town(x = {self.x}, y = {self.y}, county = {self.county})'
    
    # Returns the distance between this town and another town. Note that this is
    # symmetric between the same two towns. If the two counties are not in the
    # same county, an extra distance of county_boarder_penalty is applied.
    def distance_to(self, other):
        delta_x = self.x - other.x
        delta_y = self.y - other.y
        dist = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if self.county != other.county:
            dist += county_boarder_penalty
        return dist
    
# Helper function to parse towns from the .xls file into a usable format.
# Because the input .xls file is already sorted by county, the resulting list
# is sorted by county as well.
def parse_towns():
    # Read .xls file into pandas dataframe
    file = r'CAtowns.xls'
    df = pd.read_excel(file)
    # Add towns to a list of Town objects
    towns = []
    for row in df.itertuples():
        x_corr = row[1]
        y_corr = row[2]
        county = row[6]
        new_town = Town(x_corr, y_corr, county)
        towns.append(new_town)
    return towns

# Helper function that returns a list indicating all of the towns that are in
# a given county (a list of lists of ints, where the ints are indices into the
# list of towns). This assumes that the input list of towns is sorted by county.
def get_county_list(towns):
    counties = [[]] # Empty list in 0th index so we can index with county no
    last_county = 1 # Starts at county 1
    curr_county_list = []
    for t in range(len(towns)):
        town = towns[t]
        if town.county != last_county:
            counties.append(curr_county_list)
            curr_county_list = []
        curr_county_list.append(t)
    return counties

# Helper function to validate a given tour; that is, check that it starts and
# ends at the same town (town 0, arbitrarily), visits each other town exactly 
# once, and visits every town in a county before leaving it.
def validate_tour(towns, tour):
    num_towns = len(towns)
    counties = get_county_list(towns)
    if tour[0] != 0 or tour[-1] != 0:
        print("Invalid tour: does not start and end at 0.")
        return False
    visited = [False] * num_towns
    for s in range(len(tour) - 1):
        t = tour[s]
        if visited[t]:
            print("Invalid tour: visits the same town twice.")
            return False
        visited[t] = True
        curr_county = towns[tour[s]].county
        next_county = towns[tour[s + 1]].county
        if curr_county != next_county:
            # Check that we have visited all towns in the current county
            for i in counties[curr_county]:
                if not visited[i]:
                    print("Invalid tour: leaves county prematurely.")
                    return False        
    for v in visited:
        if not v:
            print("Invalid tour: not all towns were visited.")
            return False
    return True

# Helper function to print the order that a given tour visits each county. This
# is for fun to see how much the order of the final tour differs from that of
# the initial tour.
def print_county_order(towns, tour):
    last_county = -1
    for t in tour:
        curr_county = towns[t].county
        if curr_county != last_county:
            print("Visiting county", curr_county)
            last_county = curr_county

# Helper function to create an initial tour through all towns.
# The input list of towns is assumed to be sorted by county (as is produced by
# parse_towns). Thus, we arbitrarily start at the first town in the list, then
# follow the tour created by the order of the list of towns. The tour is
# represented as a list of indices into the provided list.
def create_initial_tour(towns):
    return list(range(len(towns))) + [0]

# Helper function that returns the total length of a given tour. The result is
# normalized to remove the cost of boarders that we cannot avoid crossing (i.e.,
# this is the true cost of the path).
def get_tour_cost(towns, tour):
    total_cost = 0
    for i in range(len(tour) - 1):
        curr = tour[i]
        nxt = tour[i + 1]
        t1 = towns[curr]
        t2 = towns[nxt]
        dst = t1.distance_to(t2)
        total_cost += dst
    # Take out required distance to visit all towns
    total_cost -= county_boarder_penalty * num_counties
    return total_cost

# Helper function that returns the cost of the partial tour. This is used to
# find the difference in costs between two possible tours without computing
# sections that are the same between the tours. start and end are indices into
# the tour list. The extra cost from crossing necessary boarders is not
# subtraced since it is removed when looking at differences in tour costs.
# This is O(n), since we are expected to need to look at 1/4 of the tour.
def get_partial_tour_cost(towns, tour, start, end):
    total_cost = 0
    for i in range(start, end):
        curr = tour[i]
        nxt = tour[i + 1]
        t1 = towns[curr]
        t2 = towns[nxt]
        dst = t1.distance_to(t2)
        total_cost += dst
    return total_cost

# Helper function that returns the difference in tour cost as a result of
# applying transition matrix Q(3) from the problem set on the tour with
# i = start_rev and j = end_rev. This is more efficient than using
# get_partial_tour_cost because it is O(1) instead of O(n).
def get_change_rev_tour_cost(towns, tour, start_rev, end_rev):
    diff = 0.0
    # Add cost of current tour
    diff += towns[tour[start_rev - 1]].distance_to(towns[tour[start_rev]])
    diff += towns[tour[end_rev]].distance_to(towns[tour[end_rev + 1]])
    # Subtract cost of resulting tour
    diff -= towns[tour[start_rev - 1]].distance_to(towns[tour[end_rev]])
    diff -= towns[tour[start_rev]].distance_to(towns[tour[end_rev + 1]])
    # All other costs are the same
    return diff

# Returns the modified tour obtained by reversing the path between start and
# end. Assuming start and end are randomly selected from the correct range, this
# is equivilent to transition matrix Q(3) from parts a/b of the problem set.
def get_modified_tour_reverse(tour, start, end):
    mod_tour = list(tour)
    n_flips = end - start + 1
    for t in range(n_flips):
        mod_tour[start + t] = tour[end - t]
    return mod_tour  

# Funtion to run the simulated annealing algorithm
def run_simulated_annealing(towns, init_tour, start_temp, num_steps):
    num_towns = len(towns)
    while True: # Repeat until we get a valid path
        tour = list(init_tour)
        # Start with n = 2 so that log(n) is defined and non-zero
        for n in range(2, num_steps + 2):
            Tn = start_temp / math.log(n)
            i = rand.randint(1, num_towns - 1)
            while True:
                j = rand.randint(1, num_towns - 1)
                if j != i:
                    break
            start_rev = min(i, j)
            end_rev = max(i, j)
            delta_cost = get_change_rev_tour_cost(towns, tour, start_rev, end_rev)
            try:
                u = math.exp(delta_cost / Tn)
                h_u = u / (u + 1)
                P = h_u
            except OverflowError as err:
                # Input to math.exp() was too large and caused overflow (this seems
                # to be happening with u on the order of e^(1000)). With this large
                # of an exponent, h_u is essentially 1, so we just set P = 1.
                P = 1
            U = rand.random()
            if U < P:
                tour = get_modified_tour_reverse(tour, start_rev, end_rev)
        # Check that the final tour is valid. If it is not, we re-run the
        # algorithm again to get a new tour. As explained in part e, the odds of
        # this happening with proper parameters are low, but it is possible that
        # we get trapped a valley that is not a valid tour. With the parameters
        # given, this happens occasionally but not too often.
        if validate_tour(towns, tour):
            break
    return tour

# Parse town data and create the initial tour        
town_list = parse_towns()
init_tour = create_initial_tour(town_list)

# Algorithm parameters. The simulated annealing function runs for num_steps
# and divides the init_temp by log(n) to get the temperature at the nth step.
init_temp = 0.1
num_steps = 100000

# Run simulated annealing algorithm to select a tour
final_tour = run_simulated_annealing(town_list, init_tour, init_temp, num_steps)
# Output the initial and final tour costs
init_cost = get_tour_cost(town_list, init_tour)
final_cost = get_tour_cost(town_list, final_tour)
print_county_order(town_list, final_tour)
print("Initial tour cost:", init_cost)
print("Final tour cost:", final_cost)
pct_improvement = (init_cost - final_cost) / init_cost * 100
print("Percent cost improvement:", pct_improvement, "%")