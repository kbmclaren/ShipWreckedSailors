# File: bayes.py
# Author: Caleb M. McLaren with contributions by Lee Vaughan
# email: Dec. 7th, 2022
# description: This file must apply Baye's Theorem to the missing sailor problem

import sys
import random
import itertools
import numpy as np
import cv2 as cv

# Assign constants (not trully immutable, safer to place in a separate file)
MAP_FILE = '../resources/cape_python.png'

# SA => Search Area 
SA1_CORNERS = (130, 265, 180, 315) # (Upper Left-X, UL-Y, Lower Right-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305) # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255) # (UL-X, UL-Y, LR-X, LR-Y)

class Search():

    def __init__(self, name):
        self.name = name
        self.img = cv.imread( MAP_FILE, cv.IMREAD_COLOR) #MAP_FILE is greyscale, but IMREAD_COLOR sets you up to use color on the self.img attribute.
        #Easy check for the fun of it.
        cv.imshow('map',self.img)
        cv.waitKey(0)
        cv.destroyAllWindows() # See [ https://www.geeksforgeeks.org/reading-image-opencv-using-python/?ref=lbp ]
        #Custom error message bc Default error message is confusing.
        if self.img is None:
            print(f"Could not load map file {MAP_FILE}.", file=sys.stderr)
            sys.exit(1)

        #Attributes for sailor's actual location when found.
        self.area_actual = 0 # "Searh area number" cause we will number the target search areas as part of the search.
        self.sailor_actual = [0,0] #"Local" (?"relative"?) Coordinates within search area.

        # Search Area are Sub-arrays within the array that is the self.img
        # self.img[ y1 : y2, x1 : x2] is a numpy convention. 
        self.sa1 = self.img[ SA1_CORNERS[1] : SA1_CORNERS[3], 
                                SA1_CORNERS[0] : SA1_CORNERS[2]]
        self.sa2 = self.img[ SA2_CORNERS[1] : SA2_CORNERS[3], 
                                SA2_CORNERS[0] : SA2_CORNERS[2]]
        self.sa3 = self.img[ SA3_CORNERS[1] : SA3_CORNERS[3], 
                                SA3_CORNERS[0] : SA3_CORNERS[2]]

        #Priors, i.e. probability we find the sailor in areas 1-3 before we start searching. Must sum to 1. 
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3
        
        if ( self.p1 + self.p2 + self.p3 ) != 1.0 :
            print(f"Prior probabilites do not sum to 100%, please revise at {self.name}", file=sys.stderr)
            sys.exit(1)

        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

        