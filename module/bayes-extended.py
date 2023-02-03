# File: bayes-extended.py
# Author: Lee Vaughan with extensions by Caleb M. McLaren
# email: kbmclaren@gmail.com
# Start date: Dec. 7th, 2022
# Last Update: Feb 2, 2023
# description: This file must apply Baye's Theorem to the missing sailor problem,
# and extend Lee Vaughn's open source game/solution, as found in his book "Real World Python".

from sys import exit, stderr
from random import triangular, uniform, shuffle
from itertools import product
from numpy import random, ndarray
from cv2 import (
    imread,
    imshow,
    waitKey,
    line,
    putText,
    rectangle,
    moveWindow,
    circle,
    IMREAD_COLOR,
    FONT_HERSHEY_PLAIN,
)  # , destroyAllWindows
from os import path


def set_map_const(resource_rel_path: str) -> str:
    """Takes a relative path for a resource and returns the absolute path for that resource."""

    if path.exists(resource_rel_path):
        abs_path = path.commonpath([__file__, resource_rel_path])
        combo = path.join(abs_path, resource_rel_path)
    else:
        raise FileNotFoundError

    if path.exists(combo):
        if isinstance(combo, str):
            return combo
        else:
            raise TypeError
    else:
        raise FileNotFoundError


rel_path = "resources/cape_python.png"
MAP_FILE = set_map_const(rel_path)

# SA => Search Area, 50 x 50 pixels in size.
SA1_CORNERS = (130, 265, 180, 315)  # (Upper Left-X, UL-Y, Lower Right-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305)  # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255)  # (UL-X, UL-Y, LR-X, LR-Y)


class Search:
    def __init__(self, name):
        self.name = name
        self.img = imread(
            MAP_FILE, IMREAD_COLOR
        )  # IMREAD_COLOR sets you up to use color indicators on map legend.

        # Sailor's location to be set by individual instance via sailor_final_location().
        self.area_actual = 0  # search area
        self.sailor_actual = [
            0,
            0,
        ]  # "Local" (?"relative"?) Coordinates within search area.

        # Custom error message bc Default error message is confusing.
        if self.img is None:
            print(f"Could not load map file {MAP_FILE}.", file=stderr)
            exit(1)

        # Search Area are Sub-arrays within the array that is the self.img
        # self.img[ y1 : y2, x1 : x2] is a numpy convention.
        self.sa1 = self.img[
            SA1_CORNERS[1] : SA1_CORNERS[3], SA1_CORNERS[0] : SA1_CORNERS[2]
        ]
        self.sa2 = self.img[
            SA2_CORNERS[1] : SA2_CORNERS[3], SA2_CORNERS[0] : SA2_CORNERS[2]
        ]
        self.sa3 = self.img[
            SA3_CORNERS[1] : SA3_CORNERS[3], SA3_CORNERS[0] : SA3_CORNERS[2]
        ]

        # Priors, i.e. probability we find the sailor in areas 1-3 before we start searching. Must sum to 1.
        # In a real life search for sailors lost at sea, these probabilites would come from the SAROPS program.
        # Future iterations of this method will need to allow for updating/randomizing these probs at start up.
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

        # sep => search effectiveness probability, which seems a misnomer since it is used more like a weight.
        # Initially set by calc_search_effectiveness()
        # Then updated indirectly by results of conduct_search() inside the "choose_number()" helper functions below.
        # The choose_number() math end up resetting sep values to zero ...
        # which works because of the updating revise_target_probablity() function needs zeros as the default sep1-3 value
        # so the important probability(p1-3) of finding the sailor doesn't change when you don't search the area.
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

    def draw_map(self, last_known: tuple) -> None:
        """draw_map() takes in the last_known coordinates of the lost sailor and draws the search map"""

        # Overlay Scale indicator
        line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)
        putText(self.img, "0", (8, 370), FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        putText(
            self.img, "50 Nautical Miles", (71, 370), FONT_HERSHEY_PLAIN, 1, (0, 0, 0)
        )

        # Draw the three search areas as rectangles
        rectangle(
            self.img,
            (SA1_CORNERS[0], SA1_CORNERS[1]),
            (SA1_CORNERS[2], SA1_CORNERS[3]),
            (0, 0, 0),
            1,
        )
        putText(
            self.img,
            "1",
            (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15),
            FONT_HERSHEY_PLAIN,
            1,
            0,
        )

        rectangle(
            self.img,
            (SA2_CORNERS[0], SA2_CORNERS[1]),
            (SA2_CORNERS[2], SA2_CORNERS[3]),
            (0, 0, 0),
            1,
        )
        putText(
            self.img,
            "2",
            (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15),
            FONT_HERSHEY_PLAIN,
            1,
            0,
        )

        rectangle(
            self.img,
            (SA3_CORNERS[0], SA3_CORNERS[1]),
            (SA3_CORNERS[2], SA3_CORNERS[3]),
            (0, 0, 0),
            1,
        )
        putText(
            self.img,
            "3",
            (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15),
            FONT_HERSHEY_PLAIN,
            1,
            0,
        )

        # Draw the legend on the map, Red "+" for last_known, and Blue "*" for actual pos.
        # openCV uses a Blue-Green-Red color format.
        putText(
            self.img, "+", (last_known), FONT_HERSHEY_PLAIN, 1, (0, 0, 255)
        )  
        putText(
            self.img,
            "+ = Last Known Position",
            (274, 355),
            FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
        )
        putText(
            self.img,
            "* = Actual Position",
            (275, 370),
            FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
        )
        imshow("Search Area", self.img)
        # This moves the image window to the top right so as to interfere with your interpreter window less.
        moveWindow(
            "Search Area", 750, 10
        )  
        waitKey(500)

    def sailor_final_location(self, num_search_areas: int) -> tuple:
        """sailor_final_location() takes in the number of search areas and returns the static x, y location of the missing sailors"""

        # Find sailor coordinates with respect to any Search Array subarray.
        # "python np.shape(self.SA1)" -> (50,50,3)
        self.sailor_actual[0] = random.choice(
            self.sa1.shape[1], 1
        )  # shape[1] chooses columns, 1 chooses a single element.
        self.sailor_actual[1] = random.choice(
            self.sa1.shape[0], 1
        )  # shape[0] chooses rows, 1 chooses a single element.

        # Randomly select one of the search areas as the search area the lost sailor is actually in.
        """
        The triangular distribution is typically used as a subjective description of a population for which there is only limited sample data, 
        and especially in cases where the relationship between variables is known but data is scarce (possibly because of the high cost of collection). 
        It is based on a knowledge of the minimum and maximum and an "inspired guess"[3] as to the modal value. 
        For these reasons, the triangle distribution has been called a "lack of knowledge" distribution."""
        area = int(triangular(1, num_search_areas + 1))

        # Note that this variable "area" is a local variable in Python and is not accessible to other methods in class Search.
        if area == 1:
            # sailor_actual[0/1] will hold a value of 0 - 49.
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

    def calc_search_effectiveness(self) -> None:
        """Set Decimal search effectiveness value per search area."""

        self.sep1 = uniform(0.2, 0.9)
        self.sep2 = uniform(0.2, 0.9)
        self.sep3 = uniform(0.2, 0.9)

    def conduct_search(
        self, area_num: int, area_array: ndarray, effectiveness_prob: float
    ) -> tuple:
        """Return search results and list of searched coordinates."""

        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])

        # Make a 2-D array of search coordinates, put coords in list, and shuffle order.
        coords = list(product(local_x_range, local_y_range))
        shuffle(coords)

        # Shrink coords list to only search as much area as we can effectively search.
        # Recall that we are operating with a search effectiveness modifier,
        # where a stormy sea reduces how much of an area we can effectively search.
        # If L is a list, the expression L [ start : stop : step ] returns the portion of the list from index start to index stop, at a step size step.
        coords = coords[: int((len(coords) * effectiveness_prob))]

        # Copy predetermined location of target to local variable.
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])

        # Search for match between area we could search and the actual location.
        if area_num == self.area_actual and loc_actual in coords:
            return (f"Found in Area {area_num}", coords)
        else:
            return ("Not Found", coords)

    def revise_target_probs(self) -> None:
        """Update search area(s) probability of finding sailor, based on search effectivness."""

        """The mechanism of update is most obvious when one of the sep values is 1 (read 100% effective).
        (1 - sep) means that if you were able to search 100% of the target area and still did not find the target, 
        that target area probability drops to zero."""
        denom = (
            (self.p1 * (1 - self.sep1))
            + (self.p2 * (1 - self.sep2))
            + (self.p3 * (1 - self.sep3))
        )

        # When you don't find the sailor, update your prediction about where the sailor will be found, OOP style.
        self.p1 = (self.p1 * (1 - self.sep1)) / denom
        self.p2 = (self.p2 * (1 - self.sep2)) / denom
        self.p3 = (self.p3 * (1 - self.sep3)) / denom


def draw_menu(search_num: int) -> None:
    """Print menu of choices for conducting area searches."""

    print(f"\nSearch {search_num + 1}")
    print(
        """
            Choose next areas to search:

            0 - Quit
            1 - Send both search teams to Area 1
            2 - Send both search teams to Area 2
            3 - Send both search teams to Area 3
            4 - Search Areas 1 & 2
            5 - Search Areas 1 & 3
            6 - Search Areas 2 & 3 
            7 - Start Over
            """
    )


def choose_zero() -> None:
    """Quit game."""
    exit()


def choose_one(SearchObject: Search) -> tuple:
    """Send both search teams to Area 1, return results and coordinates."""
    results_1, coords_1 = SearchObject.conduct_search(
        1, SearchObject.sa1, SearchObject.sep1
    )
    results_2, coords_2 = SearchObject.conduct_search(
        1, SearchObject.sa1, SearchObject.sep1
    )
    # As a reminder, set() drops duplicates here.
    SearchObject.sep1 = (len(set(coords_1 + coords_2))) / (
        len(SearchObject.sa1) ** 2
    )  
    # The area was not searched so we don't want to update previous prob that sailor would be found.
    SearchObject.sep2 = 0  
    SearchObject.sep3 = 0
    return results_1, coords_1, results_2, coords_2


def choose_two(SearchObject: Search) -> tuple:
    """Send both search teams to Area 2, return results and coordinates."""
    results_1, coords_1 = SearchObject.conduct_search(
        2, SearchObject.sa2, SearchObject.sep2
    )
    results_2, coords_2 = SearchObject.conduct_search(
        2, SearchObject.sa2, SearchObject.sep2
    )
    SearchObject.sep1 = 0
    SearchObject.sep2 = (len(set(coords_1 + coords_2))) / (len(SearchObject.sa2) ** 2)
    SearchObject.sep3 = 0

    return results_1, coords_1, results_2, coords_2


def choose_three(SearchObject: Search) -> tuple:
    """Send both search teams to Area 3, return results and coordinates."""
    results_1, coords_1 = SearchObject.conduct_search(
        3, SearchObject.sa3, SearchObject.sep3
    )
    results_2, coords_2 = SearchObject.conduct_search(
        3, SearchObject.sa3, SearchObject.sep3
    )
    SearchObject.sep1 = 0
    SearchObject.sep2 = 0
    SearchObject.sep3 = (len(set(coords_1 + coords_2))) / (len(SearchObject.sa3) ** 2)
    return results_1, coords_1, results_2, coords_2


def chooseFour(SearchObject: Search) -> tuple:
    """Search Areas 1 & 2, return results and coordinates."""
    results_1, coords_1 = SearchObject.conduct_search(
        1, SearchObject.sa1, SearchObject.sep1
    )
    results_2, coords_2 = SearchObject.conduct_search(
        2, SearchObject.sa2, SearchObject.sep2
    )
    SearchObject.sep3 = 0
    return results_1, coords_1, results_2, coords_2


def choose_five(SearchObject: Search) -> tuple:
    """Search Areas 1 & 3, return results and coordinates."""
    results_1, coords_1 = SearchObject.conduct_search(
        1, SearchObject.sa1, SearchObject.sep1
    )
    results_2, coords_2 = SearchObject.conduct_search(
        3, SearchObject.sa3, SearchObject.sep3
    )
    SearchObject.sep2 = 0
    return results_1, coords_1, results_2, coords_2


def choose_six(SearchObject: Search) -> tuple:
    """Search Areas 2 & 3, return results and coordinates."""
    results_1, coords_1 = SearchObject.conduct_search(
        2, SearchObject.sa2, SearchObject.sep2
    )
    results_2, coords_2 = SearchObject.conduct_search(
        3, SearchObject.sa3, SearchObject.sep3
    )
    SearchObject.sep1 = 0
    return results_1, coords_1, results_2, coords_2


def choose_seven() -> None:
    """Calls main() to start a new game."""
    main()


def choose_invalid() -> None:
    """Provides feedback to disapproved user input."""
    print("\nSorry, but that isn't a valid choice.", file=stderr)


def set_hurricane_arrival() -> int:
    """Simulating an approaching hurricane, Returns number of rounds the player has to find the sailor before a forced restart of game."""
    searchLimit = uniform(3, 9)
    return int(searchLimit)


def main():
    app = Search("Cape_Python")

    # This next bit annoys me and I want to rewrite to accept user input. But my purpose is to read the book so I'll skip for now. (https://pynative.com/python-check-user-input-is-number-or-string/)
    app.draw_map(last_known=(160, 290))
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)

    # cv.circle did not accept ndarrays, convert to python integers.
    sailor_x = sailor_x.item()
    sailor_y = sailor_y.item()

    # Display game header
    print("#" * 66)
    print("-" * 28, "NEW GAME", "-" * 28)
    print("#" * 66)
    print("\nInitial Target (P) Probabilities:")
    print(f"P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}")

    # Before main game loop, set max number of turns before hurricane stops game.
    search_num = 0
    search_limit = set_hurricane_arrival()

    # While loop stopped by the following: break statement, recursive call to main, and sys.exit.
    while True:
        # Set effectiveness randomly to simulate variable sea conditions
        app.calc_search_effectiveness()
        draw_menu(search_num)

        # Unable to use match-case structure since restricted to python 3.8.5 ... maybe.
        # Use dictionary alternative to long, unreadable if, elif, else structure.
        choiceDict = {
            "0": choose_zero,
            "1": choose_one,
            "2": choose_two,
            "3": choose_three,
            "4": chooseFour,
            "5": choose_five,
            "6": choose_six,
            "7": choose_seven,  # recursive call to main(
        }

        choice = input("Choice: ")

        # Using the choice Dictionary and helper functions above make this evalauation structure 6 elifs smaller.
        if choice not in choiceDict:
            # handle incorrect input
            choose_invalid()

            # continue skips the rest of the while loop, so holdMyTuple does not get evaluated.
            continue

        elif choice == "0":
            # end game
            choose_zero()

        elif choice == "7":
            # start new game
            choose_seven()

        else:
            # return helper function to variable and call helper function by alternate name "search_settings_by_choice"
            search_settings_by_choice = choiceDict.get(choice)
            holdMyTuple = search_settings_by_choice(app)

        # Update predictive probablity that sailor will be found in search areas.
        app.revise_target_probs()

        print(f"\nSearch {search_num + 1} Effectiveness (E): ")
        print(f"E1 = {app.sep1:.3f}, E2 = {app.sep2:.3f}, E3 = {app.sep3:.3f}")
        print(f"\nSearch {search_num + 1} Results 1 = {holdMyTuple[0]}", file=stderr)
        print(f"Search {search_num + 1} Results 2 = {holdMyTuple[2]}", file=stderr)
        print(f"#" * 65)

        # Recall, holdMyTuple = (results_1, coords_1, results_2, coords_2), ...
        if holdMyTuple[0] == "Not Found" and holdMyTuple[2] == "Not Found":
            search_num += 1
            if search_num == search_limit:
                # Skip another eval of while loop condition.
                break

            print(f"\nNew Target Probabilities (P) for Search {search_num + 2}:")
            print(f"P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}")

        else:
            # Negative thickness fills in the circle with color.
            # cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
            print("To continue: Left click on game map, and press enter on keyboard.")
            circle(app.img, (sailor_x, sailor_y), 3, (255, 0, 0), -1)
            imshow("Search Area", app.img)
            waitKey(0)
            main()

    print(
        f"""
    The sailor could not be recovered before a hurricane forced the search to end.
    You made {search_num} searches before the hurricane arrived."""
    )
    main()


if __name__ == "__main__":
    main()
