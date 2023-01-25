# File: bayes.py
# Author: Caleb M. McLaren with contributions by Lee Vaughan
# email: kbmclaren@gmail.com
# Start date: Dec. 7th, 2022
# Last Update: Jan 19, 2023
# description: This file must apply Baye's Theorem to the missing sailor problem

import sys
import random
import itertools
import numpy as np
import cv2 as cv

# Assign constants (not trully immutable, safer to place in a separate file)
MAP_FILE = '../resources/cape_python.png'

# SA => Search Area, 50 x 50 pixels in size.
SA1_CORNERS = (130, 265, 180, 315) # (Upper Left-X, UL-Y, Lower Right-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305) # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255) # (UL-X, UL-Y, LR-X, LR-Y)

def draw_menu(search_num):
        """Print menu of choices for conducting area searches."""
        print(f'/nSearch {search_num}')
        print(
            """
            Choose next areas to search:

            0 - Quit
            1 - Search Area 1 twice over
            2 - Search Areas 2 twice over
            3 - Search Area 3 twice over
            4 - Search Areas 1 & 2
            5 - Search Areas 1 & 3
            6 - Search Areas 2 & 3 
            7 - Start Over
            """
            )

class Search():

    def __init__(self, name):
        self.name = name
        self.img = cv.imread( MAP_FILE, cv.IMREAD_COLOR) #MAP_FILE is greyscale, but IMREAD_COLOR sets you up to use color on the self.img attribute.
         #Attributes for sailor's actual location when found.
        self.area_actual = 0 # "Search area number" because we will number the target search areas as part of the search.
        self.sailor_actual = [0,0] #"Local" (?"relative"?) Coordinates within search area.
        
        #Easy check for the fun of it.
        cv.imshow('map', self.img)
        cv.waitKey(0)
        cv.destroyAllWindows() # See [ https://www.geeksforgeeks.org/reading-image-opencv-using-python/?ref=lbp ]
        
        #Custom error message bc Default error message is confusing.
        if self.img is None:
            print(f"Could not load map file {MAP_FILE}.", file=sys.stderr)
            sys.exit(1)

        # Search Area are Sub-arrays within the array that is the self.img
        # self.img[ y1 : y2, x1 : x2] is a numpy convention. 
        self.sa1 = self.img[ SA1_CORNERS[1] : SA1_CORNERS[3], 
                                SA1_CORNERS[0] : SA1_CORNERS[2]]
        self.sa2 = self.img[ SA2_CORNERS[1] : SA2_CORNERS[3], 
                                SA2_CORNERS[0] : SA2_CORNERS[2]]
        self.sa3 = self.img[ SA3_CORNERS[1] : SA3_CORNERS[3], 
                                SA3_CORNERS[0] : SA3_CORNERS[2]]

        #Priors, i.e. probability we find the sailor in areas 1-3 before we start searching. Must sum to 1.
        # In a real life search for sailors lost at sea, these probabilites would come from the SAROPS program. 
        # Future iterations of this method will need to allow for updating these probs at start up.  
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

        # Initially set by calc_search_effectiveness()
        # Then updated indirectly by results of conduct_search() inside the "choose_()" helper functions below.
        # The choose_() math end up resetting sep values to zero ... 
        # which works because of the updating revise_target_probablity function needs zeros as the default sep1-3 value 
        # so the important probability(p1-3) of finding the sailor doesn't change when you don't search the area.
        # sep => search effectiveness probability, which seems a misnomer since it is used more like a weight.
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

    def draw_map(self, last_known):
        """draw_map() takes in the last_known coordinates of the lost sailor"""

        #Overlay Scale indicator
        cv.line(self.img, (20, 370), (70, 370), (0,0,0), 2)
        cv.putText(self.img, '0', (8,370), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(self.img, '50 Nautical Miles', (71,370), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0)) 

        #Draw the three search areas as rectangles
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), 
                                (SA1_CORNERS[2], SA1_CORNERS[3]), (0,0,0), 1)
        cv.putText(self.img, '1', (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]), 
                                (SA2_CORNERS[2], SA2_CORNERS[3]), (0,0,0), 1)
        cv.putText(self.img, '2', (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]), 
                                (SA3_CORNERS[2], SA3_CORNERS[3]), (0,0,0), 1)
        cv.putText(self.img, '3', (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        #Draw the legend on the map
        # Red "+" for last_known, 
        # Blue "*" for actual pos. 
        cv.putText(self.img, '+', (last_known), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255)) # openCV uses a Blue-Green-Red color format.
        cv.putText(self.img, '+ = Last Known Position', (274, 355), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '* = Actual Position', (275, 370), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 750, 10) #This moves the image window to the top right so as to infere with your interpreter window less.
        cv.waitKey(500)

    def sailor_final_location(self, num_search_areas): 
        """sailor_final_location() takes in the number of search areas and returns the actual x, y location of the missing sailors""" 
        """because this is a game where we need to set the answer before the game can begin.
        I'm a little suprised this method isn't called in __init__... but maybe I'am not thinking in a Pythonic style.
        An improved version of this game would simulate how a sailor would change position over time, 
        so the current static position set by this method is more like setting a location of a sunken ship or submarine."""
        
        #Find sailor coordinates with respect to any Search Array subarray.
        # np.shape(self.SA1) -> (50,50,3)
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1], 1) # shape[1] chooses columns, 1 chooses a single element. 
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0], 1) # shape[0] chooses rows, 1 chooses a single element.

        #Randomly select one of the search areas as the search area the lost sailor is actually in.
        """The triangular distribution is typically used as a subjective description of a population for which there is only limited sample data, 
        and especially in cases where the relationship between variables is known but data is scarce (possibly because of the high cost of collection). 
        It is based on a knowledge of the minimum and maximum and an "inspired guess"[3] as to the modal value. 
        For these reasons, the triangle distribution has been called a "lack of knowledge" distribution.""" 
        area = int(random.triangular(1, num_search_areas + 1))

        # Note that this variable "area" is a local variable in Python and is not accessible to other methods in class Search.
        if area == 1:
            #sailor_actual[0/1] will hold a value of 0 - 49. 
            x = self.sailor_actual[0] + SA1_CORNERS[0]
            y = self.sailor_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2: 
            x = self.sailor_actual[0] + SA2_CORNERS[0]
            y = self.sailor_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3: 
            x = self.sailor_actual[0] + SA3_CORNERS[0]
            y = self.sailor_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3
        return (x, y)

    def calc_search_effectiveness(self):
        """Set Decimal search effectiveness value per search area. 
        I get the sense that this method and the previous will be called in a script or member method to set up the game, 
        but not for game play."""
        
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def conduct_search(self, area_num, area_array, effectiveness_prob):
        """Return search results and list of searched coordinates."""
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])

        # Make a 2-D array of search coordinates, put coords in list, and shuffle order.
        coords = list(itertools.product(local_x_range, local_y_range))
        random.shuffle(coords)
        
        # Shrink coords list to only search as much area as we can effectively search. Recall that we are operating with an effectiveness modifier, 
        # where a stormy sea reduces how much of an area we can effectively search.
        # If L is a list, the expression L [ start : stop : step ] returns the portion of the list from index start to index stop, at a step size step. 
        coords = coords[:int((len(coords) * effectiveness_prob))]

        # Copy predetermined location of target to local variable.
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])

        # Search for match between area we could search and the actual location. 
        if area_num == self.area_actual and loc_actual in coords:
            return f"Found in Area {area_num}" , coords 
        else:
            return "Not Found", coords

    def revise_target_probs(self):
        """Update search area(s) probability of finding sailor, based on search effectivness."""
        """The mechanism of update is most obvious when one of the sep values is 1 (read 100% effective).
        (1 - sep) means that if you were able to search 100% of the target area and still did not find the target, 
        that target area probability drops to zero."""
        denom = (self.p1 * (1 - self.sep1)) + (self.p2 * (1 - self.sep2)) + (self.p3 * (1 - self.sep3))

        #When you don't find the sailor, update your prediction about where the sailor will be found, OOP style.
        self.p1 = (self.p1 * (1 - self.sep1))/denom
        self.p2 = (self.p2 * (1 - self.sep2))/denom
        self.p3 = (self.p3 * (1 - self.sep3))/denom
        
def chooseZero():
    sys.exit()

def chooseOne( SearchObject: Search) -> tuple:
    results_1, coords_1 = SearchObject.conduct_search(1, SearchObject.sa1, SearchObject.sep1)
    results_2, coords_2 = SearchObject.conduct_search(1, SearchObject.sa1, SearchObject.sep1)
    SearchObject.sep1 = (len(set(coords_1 + coords_2))) / (len(SearchObject.sa1)**2) #As a reminder, set() drops duplicates here.
    SearchObject.sep2 = 0 #The area was not searched so we don't want to update previous prob that sailor would be found. 
    SearchObject.sep3 = 0
    #return ((results_1, coords_1, results_2, coords_2 ), SearchObject.sep1, SearchObject.sep2, SearchObject.sep3)
    return results_1, coords_1, results_2, coords_2

def chooseTwo( SearchObject: Search) -> tuple:
    results_1, coords_1 = SearchObject.conduct_search(2, SearchObject.sa2, SearchObject.sep2)
    results_2, coords_2 = SearchObject.conduct_search(2, SearchObject.sa2, SearchObject.sep2)
    SearchObject.sep1 = 0
    SearchObject.sep2 = (len(set(coords_1 + coords_2))) / (len(SearchObject.sa2)**2)
    SearchObject.sep3 = 0
    #return ((results_1, coords_1, results_2, coords_2 ), SearchObject.sep1, SearchObject.sep2, SearchObject.sep3)
    return results_1, coords_1, results_2, coords_2

def chooseThree( SearchObject: Search) -> tuple:
    results_1, coords_1 = SearchObject.conduct_search(3, SearchObject.sa3, SearchObject.sep3)
    results_2, coords_2 = SearchObject.conduct_search(3, SearchObject.sa3, SearchObject.sep3)
    SearchObject.sep1 = 0
    SearchObject.sep2 = 0
    SearchObject.sep3 = (len(set(coords_1 + coords_2))) / (len(SearchObject.sa3)**2)
    #return ((results_1, coords_1, results_2, coords_2 ), SearchObject.sep1, SearchObject.sep2, SearchObject.sep3)
    return results_1, coords_1, results_2, coords_2

def chooseFour( SearchObject: Search) -> tuple:
    results_1, coords_1 = SearchObject.conduct_search(1, SearchObject.sa1, SearchObject.sep1)
    results_2, coords_2 = SearchObject.conduct_search(2, SearchObject.sa2, SearchObject.sep2)
    SearchObject.sep3 = 0
    #return ((results_1, coords_1, results_2, coords_2), SearchObject.sep1, SearchObject.sep2, SearchObject.sep3)
    return results_1, coords_1, results_2, coords_2

def chooseFive( SearchObject: Search) -> tuple:
    results_1, coords_1 = SearchObject.conduct_search(1, SearchObject.sa1, SearchObject.sep1)
    results_2, coords_2 = SearchObject.conduct_search(3, SearchObject.sa3, SearchObject.sep3)
    SearchObject.sep2 = 0
    #return ((results_1, coords_1, results_2, coords_2), SearchObject.sep1, SearchObject.sep2, SearchObject.sep3)
    return results_1, coords_1, results_2, coords_2
    
def chooseSix( SearchObject: Search) -> tuple:
    results_1, coords_1 = SearchObject.conduct_search(2, SearchObject.sa2, SearchObject.sep2)
    results_2, coords_2 = SearchObject.conduct_search(3, SearchObject.sa3, SearchObject.sep3)
    SearchObject.sep1 = 0
    #return ((results_1, coords_1, results_2, coords_2), SearchObject.sep1, SearchObject.sep2, SearchObject.sep3,)
    return results_1, coords_1, results_2, coords_2

def chooseSeven():
    """I need to better understand what happens to the original app object."""
    main()

def chooseInvalid():
    print("/nSorry, but that isn't a valid choice.", file=sys.stderr)

def main(): 
    """This function sets up the game and feeds the Search object the required data for the self.variables/game set up."""
    app = Search('Cape_Python')
    # This next bit annoys me and I want to rewrite to accept user input. But my purpose is to read the book so I'll skip for now. (https://pynative.com/python-check-user-input-is-number-or-string/)
    app.draw_map(last_known=(160,290))
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)
    print("-" * 65)
    print("\nInitial Target (P) Probabilities:")
    print(f"P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}")
    
    search_num = 1
    while True: 
        app.calc_search_effectiveness() # set randomly to simulate variable sea conditions
        draw_menu(search_num)

        #Unable to use match-case structure since restricted to python 3.8.5 ... maybe.
        #Use alternative to long, unreadable if, elif, else structure.
        choiceDict = {
            "0": chooseZero,
            "1": chooseOne,
            "2": chooseTwo,
            "3": chooseThree,
            "4": chooseFour,
            "5": chooseFive, 
            "6": chooseSix,
            "7": chooseSeven #recursive call to main()
        }

        choice = input("Choice: ")
        if choice in choiceDict and choice != "7": 
            search_settings_by_choice = choiceDict.get(choice)
            # Recall that python does not do pass by value/reference, but by assignment. 
            # So reassign any value changed by the helper funciton.
            # Not necessary, since the helper functions accept the Search instance and can do work directly on the instance member variables.
            # holdMyTuple, app.sep1, app.sep2, app.sep3 = search_settings_by_choice(app) 
            holdMyTuple = search_settings_by_choice(app)

        elif choice in choiceDict and choice == "7":
            #search_settings_by_choice = choiceDict.get(choice)
            #search_settings_by_choice(app)  would be usefull if main() checked for existing Search object and skipped creation if provided.
            chooseSeven()
            holdMyTuple = None #unreachable. Leave it, just in case. Cause I'm superstitious.

        else:
            chooseInvalid()
            holdMyTuple = None
            continue

        app.revise_target_probs()

        print(f"/nSearch {search_num} Results 1 = {holdMyTuple[0]}", file=sys.stderr)
        print(f"/nSearch {search_num} Results 2 = {holdMyTuple[2]}", file=sys.stderr)
        print(f"Search {search_num} Effectiveness (E): ")
        print(f"E1 = {app.sep1:.3f}, E2 = {app.sep2:.3f}, E3 = {app.sep3:.3f}")

        # Recall, holdMyTuple = (results_1, coords_1, results_2, coords_2), when it exists...
        if holdMyTuple[0] == "Not Found" and holdMyTuple[2] ==  "Not Found":
            print(f'/nNew Target Probabilities (P) for Search {search_num + 1}:')
            print(f"P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}")

        elif holdMyTuple == None:
            continue #ugh, I don't like this management of the recursive main()/continue outcomes of User choice eval.
        
        else: 
            cv.circle(app.img, (sailor_x, sailor_y), 3, (255, 0, 0), -1)
            cv.imshow('Search Area', app.img)
            cv.waitKey(1500)
            main()

        search_num += 1

if __name__ == '__main__':
    main()

    