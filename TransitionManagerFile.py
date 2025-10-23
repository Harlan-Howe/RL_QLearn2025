from MapFile import Map
from CellFile import Cell
from typing import Tuple, List
import random
Coordinates = Tuple[int, int] #row, col
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

ACTION_NAMES = ("UP","LEFT","DOWN","RIGHT")
MOVE_PENALTY = -1

class TransitionManager:
    """
    This class acts as a buffer between the Map and the Learner, since the Learner isn't supposed to know about the
    rewards directly. It has several methods that allow the Learner to find out what will happen if it attempts a
    given action, and lets the Learner calculate the "best" action from a given point.
    """
    def __init__(self, map: Map, Q_grid: List[List[int]]):
        """
        creates a new Transition object, which keeps track of both the map and the Q_grid. The learner shares the
        Q_grid, but is not allowed to directly see the map.
        :param map:
        :param Q_grid:
        """
        self.myMap = map
        self.Q_grid = Q_grid

        # a bit of shorthand for how much to change (row, col) for "Up, Right, Down, Left".
        self.deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))

        self.stochastic_levels = self.generate_probabilities()

    def generate_probabilities(self) -> Tuple[float, float, float, float]:
        """
        In an ideal world, the robot will go in whatever mode we tell it to - this would be the "deterministic"
          behavior.
        On the other hand, there could be a chance that the robot will only go in the direction selected some percentage
          of the time - and it might go in a different direction than was requested sometimes. This more random behavior
          is "stochastic" behavior.
        In this method we are setting up a variable that measures the odds that the robot will travel as follows:
            • The probability that the robot will go in the direction requested.
            • The probability that the robot will go 90° clockwise from the direction requested.
            • The probability that the robot will go 180° from the direction requested.
            • The probability that the robot will go 270° clockwise (i.e., 90° counterclockwise) from the direction
                   requested.
        :return: a tuple of four probabilities, which MUST add up to 1.0.
        """
        return (1.0, 0.0, 0.0, 0.0) #deterministic - always goes as expected
        # return (0.8, 0.1, 0, 0.1) #stochastic, going straight 80% of the time.

    def perform_action(self, s: Coordinates, a: int) -> Tuple[Coordinates, int]:
        """
        Attempt to move in the given direction, a, from location s, returning the new location, s'.
        If there is a wall in the way, then s' = s. If s' is a terminal location, set Q = reward for that location.
        :param s: starting position
        :param a: action (one of the four directions)
        :return: the new position of our robot, and whether the robot is now in a non-terminal location.
        """

        actual_a = a + self.get_action_modifier()

        new_pos = (max(0, min(self.myMap.num_cols, s[0] + self.deltas[actual_a % 4][0])),
                   max(0, min(self.myMap.num_rows, s[1] + self.deltas[actual_a % 4][1])))
        if self.myMap.get_cell_at(new_pos[0], new_pos[1]).reward <= -9000:
            s_prime = s
            self.myMap.get_cell_at(new_pos[0], new_pos[1]).visited = True
        else:
            s_prime = new_pos
        self.myMap.get_cell_at(s_prime[0], s_prime[1]).visited = True
        reward = self.myMap.get_cell_at(s_prime[0],s_prime[1]).reward
        if reward < -50 or reward > 0:
            return s_prime, reward
        else:
            return s_prime, reward+MOVE_PENALTY

    def get_best_neighbor_Q(self, s: Coordinates, a_list=(0, 1, 2, 3)) -> Tuple[float, int]:
        best_Q = -999999
        best_a = 0

        for a in a_list:
            Q_prime = self.get_expected_Q_for_action_from_pos(a, s)
            if Q_prime > best_Q:
                best_Q = Q_prime
                best_a = a
        return best_Q, best_a

    def get_actual_Q_for_action_from_pos(self, a: int, s: Coordinates) -> float:
        new_pos = (max(0, min(self.myMap.num_rows, s[0] + self.deltas[a % 4][0])),
                   max(0, min(self.myMap.num_cols, s[1] + self.deltas[a % 4][1])))
        # print(f"{s=}\t{ACTION_NAMES[a]}\t{new_pos=}")
        # print(f"{self.myMap.get_cell_at(new_pos[0], new_pos[1]).reward=}")
        if self.myMap.get_cell_at(new_pos[0], new_pos[1]).reward <= -9000:
            s_prime = s
        else:
            s_prime = new_pos
        # print(f"{s_prime=}")
        return self.Q_grid[s_prime[0]][s_prime[1]]

    def get_expected_Q_for_action_from_pos(self, a:int, s: Coordinates):
        """
        finds the expectation value for the move a from position s. Specifically, it finds the sum over
        all the directional shifts in stochastic_levels of the probability stored in that location in the
        list times the Q for the cell (s') where you would wind up if you went that way.

        Note that for the deterministic version of stochastic_levels (i.e, (1,0,0,0)) this is just one times
        Q found in the direction of the desired action, a. (Plus zero, plus zero, plus zero.)

        In contrast, if you had a stochastic_levels of (0.8, 0.1, 0, 0.1), then you would get
        total Q = 0.8 * (Q for desired action)
                + 0.1 * (Q to the right of desired action)
                + 0
                + 0.1 * (Q to the left of desired action)

        :param a: the desired action to test
        :param s: the starting location
        :return: the expected Q value
        """
        Q = 0
        for i in range(4):
            if self.stochastic_levels[i] == 0:
                continue # skip any zero-weighted values.
            Q += self.stochastic_levels[i] * self.get_actual_Q_for_action_from_pos((a+i) % 4, s)
        return Q

    def get_action_modifier(self) -> int:
        """
        gets a number from 0-3 corresponding to a difference between the desired angle moved and the actual
        angle moved. This is done with a probability established by the stochastic_levels. That is, if the
        stochastic_levels are (0.8, 0.1, 0, 0.1), then we would expect to return a zero 80% of the time, a one 10% of
        the time, a two never (0% of the time), and a three 10% of the time.
        :return:
        """
        r = random.random() # a random number from 0 ≤ r < 1
        sum_levels = 0
        for i in range(4):
            sum_levels += self.stochastic_levels[i]
            if r < sum_levels:
                return i

        raise RuntimeError(f"Problem generating action_modifier. {r=}\t{self.stochastic_levels=}\t{sum_levels=}")

