import random
import pyglet
from typing import List, Tuple, Optional
import copy
import CellFile
from TransitionManagerFile import TransitionManager


# this is the font size of the text at the top of the "world" - feel free to tinker to make it legible.
STAT_FONT_SIZE = 14

Coordinates = Tuple[int, int]

# constants - please don't change these two.
STEP_MODE_SINGLE_STEP = 0
STEP_MODE_ONE_PATH = 1


class Learner:
    """
    The Learner class is the part of the program that sends out robots to explore the world and learn about it,
    so that the next try is more successful. It develops a "Q-grid" of relative values for each navigable square
    to help with these decisions.
    """
    def __init__(self, Q_grid: List[List[float]], transition_manager: TransitionManager, start_position: Coordinates):
        """
        creates the new learner, starting with the following information
        :param Q_grid: an empty grid of zeros, the same size of the map. This is what the Learner will be updating!
        :param transition_manager: A class that knows about the map (even if the learner doesn't yet). It receives
        move commands and indicates where the robot will be now, and finds the anticipated Q value if the robot
        is told to move.
        :param start_position: The (row, col) location where all robots should start their journeys
        """
        self.Q_grid = Q_grid
        self.transition_manager = transition_manager
        self.start_coords = start_position
        self.current_coords: Optional[Coordinates] = None

        # These three constants will affect the way the robot moves and learns.
        self.learning_rate = 0.05       # alpha
        self.discount_factor = 0.85     # gamma
        self.exploration_ratio = 1.00   # epsilon

        # Some (initially) empty lists that we'll be filling in later.
        self.latest_path = []  # this is where the blue lines come in - a list of the (row, col) where the last robot
        #                         has been.
        self.current_path = []  # this is where the magenta lines come in - a list of the (row, col) where the current
        #                         robot has been.
        self.things_to_draw = []  # these are the blue lines, and text that are drawn from self.latest_pat

        self.latest_reward = 0      # What did the last robot "score" in total at the end of its journey?
        self.total_reward_for_current_run = 0     # What is the reward (so far) for the current robot?
        self.mean_reward = 0

        # a counter that we'll keep track of which robot trial we are using.
        self.run_num = 0

        self.reset_robot()
        self.should_draw_robot = True   # we're defaulting to the behavior that draws the current location of the robot
        #                                 and its current path; if we are skipping individual steps to perform one run
        #                                 per update, then we'll set this to False later.

    def reset_robot(self) -> None:
        """
        Reset the robot to the starting position and clear the paths and rewards
        :return: None
        """
        # return the robot's position to the start position.
        self.current_coords = copy.deepcopy(self.start_coords)

        # copy the current path and reward to the "latest" path and reward, then reset the "current" variables
        # for another run.
        self.latest_path = self.current_path
        self.latest_reward = self.total_reward_for_current_run

        self.current_path = [self.current_coords]
        self.total_reward_for_current_run = 0

        if self.run_num > 0:
            self.mean_reward = (self.mean_reward*(self.run_num-1) + self.latest_reward)/self.run_num

        # self.latest_action_list = []  # used for debugging only. Currently commented out.

        # increment which run this is going to be.
        self.run_num += 1

    def explore_path(self) -> None:
        """
        Creates an imaginary robot at the start location and sends it on its way through the maze until it reaches
        a terminal point (where either reward > 0 or reward < -50). Updates the values of Q along the way, so the
        next robot will learn from its victories or failures.
        :return: None
        """
        # Reset the robot to the starting position and clear the paths and rewards
        self.reset_robot()
        self.should_draw_robot = False
        # Loop as long as the robot hasn't hit a terminal cell.
        keep_going = True
        while keep_going:
            keep_going = self.do_step()

    def do_step(self) -> bool:
        """
        Perform one move by this robot. If we have hit a terminal square, reset and return False, otherwise return True.
        Updates the values of Q along the way, so the next robot will learn from its victories or failures.
        :return: True, if we are going to continue with this exploration, False if we've hit a terminal spot.
        """
        # decide on an action to take... this "action_num" will be a number corresponding to which direction you
        # want to go.
        action_num = self.select_action(self.current_coords)
        # self.latest_action_list.append(TransitionManagerFile.ACTION_NAMES[action_num])  # debugging. see line 78.

        # Make the move, based on the action selected - you get back the new location and the reward on that square.
        # Note: "perform_action" returns _two_ pieces of information (which is special to Python), so
        #       self.current_coords and reward will receive those two pieces.
        self.current_coords, reward = self.transition_manager.perform_action(self.current_coords, action_num)

        self.total_reward_for_current_run = self.total_reward_for_current_run + reward

        self.current_path.append(self.current_coords)   # We're just tracking this so we can draw the paths onscreen.
        #                                                 It's not really necessary for the algorithm to work.

        # Check whether we have hit a terminal cell--that is, one with a final reward/penalty that will make us restart.
        if reward > 0 or reward < -50:
            self.Q_grid[self.current_coords[0]][self.current_coords[1]] = reward
            return False  # We hit a terminal state, so stop this run.

        else:
            # Update the Q value for the new cell before you make your next move.
            self.update_Q(self.current_coords, reward)
            return True

    def select_action(self, position) -> int:
        """
        selects an action (0-3) to move, which might be random or based on the information stored in the Q-grid.
        The self.exploration_ratio (epsilon) gives the odds that it will be random Exploration, as opposed to
        knowledge-based Exploitation.
        :param position: the (row, col) where the robot is before the action.
        :return: a number 0-3 indicating which direction the robot should be ordered to move.
        ( 0 --> up; 1 --> right; 2 --> down; 3 --> left)
        """

        # TODO: based on self.exploration_ratio (i.e., epsilon), either return a random action or what the transition
        #  manager thinks will be the best action to take.
        #  Here are the pieces for this:
        #   "self.exploration_ratio" is the equivalent of epsilon in our discussions. It is a number 0-1; the higher it
        #                 is, the more likely we are to "explore" (i.e., go random) than "exploit" (i.e., use Q to
        #                 choose an action.)
        #   "random.random()" will pick a random number between 0-1.
        #   "self.transition_manager.get_best_neighbor_Q(position, action_list)" will ask the transition manager to
        #                 look around at the neighboring squares that are reachable by the options in action_list and
        #                 pick the one with the largest Q value. (Action_list should be a list of [0, 1, 2, 3] options,
        #                 potentially shuffled.) This method returns (best_Q, best_a) - a tuple with the highest
        #                 Q value you could get and the action (0-3) that led to it - you only need the second part.
        #                 (This method also gets called elsewhere in the program, where it needs to know the best Q,
        #                 but we don't care what the Q actually is in this method.)
        #  So you need to return _either_ a random action (0-3), inclusive; or the action that you get back from the
        #  transition manager; depending on epsilon and the random number picked. (Either way, you're returning an
        #  action.)

        return 0  # replace this!!!!

    def update_Q(self, s: Coordinates, reward: int):
        """
        Look at the value of Q for the cells around this one and find the highest expected value of Q (based on
        possible actions). Update the Q for this cell, based on that information and the value of the learning
        constant.
        :param s: where (row, col) the robot currently is - which cell to update.
        :param reward: reward for moving to this square
        :return:
        """
        # get the highest choice of Q from the transition manager.
        # (Note this method returns both the highest q and the action that gave it - we only need the former for this
        # method. The previous method also calls this method, and it cares about best_a, but it doesn't get used here.)

        best_Q, _best_a = self.transition_manager.get_best_neighbor_Q(s)  # the _ before best_a means we won't use it.
        x = s[0]
        y = s[1]

        existing_score = self.Q_grid[x][y]

        # TODO: Your job for this method is to revise the "Q" number found at self.Q_grid[x][y]. (So the last line of
        #       this method should start "self.Q_grid[x][y] = " something....
        #    Here are the parts to do this:
        #       "best_Q" is now the highest Q value you'd expect to get if you tried all four actions.
        #       "self.discount_factor" (i.e., gamma) is the multiplier we use for future scores when calculating Q.
        #       "calculated_Q" doesn't exist yet. I suggest that you make one. It should be the reward for getting here,
        #                  plus the discounted best_Q.
        #       "self.learning_rate" (i.e., alpha) is the amount that we are willing to change the existing value to
        #                  this calculated value.
        #                  So, for example, if self.learning_rate was 0.1, then we would say that the new value for the
        #                  Q_grid would be 10% based on the calculated_Q, and 90% based on the Q_grid value we had at
        #                  the start of this method. So if calculated_Q was 80 and the Q_grid at this location were 50,
        #                  the new Q_grid value would be:  0.1 * 80 + 0.9 * 50 = 53.
        #       "reward" (i.e., R) is the reward we got for moving to this square. (which my be -1 or something else.)

        # so moving one square to that space would mean losing one point for that move, but adding the score we just
        # found.

        # Now combine that "calculated" score with the existing score. Use the learning rate to combine them.



# ---------------------------------------------------  DRAWING CODE --------------------------------------------
# The rest of this file is about displaying what the robot is doing on the screen.

    def draw_self(self,
                  batch: pyglet.graphics.Batch,
                  q_group: pyglet.graphics.Group,
                  last_path_group: pyglet.graphics.Group,
                  current_path_group: pyglet.graphics.Group):
        """
        updates the Q values, the latest path, and the stats for so they are ready for the next time the page
        refreshes. (Note this is mostly delegated to three sub methods to do most of the work.)
        :param batch: the storage of the commands to draw
        :param q_group: the second layer - this will print the "Q" numbers after the cell squares
        :param last_path_group: the pyglet layer used to draw the last path
        :param current_path_group: The third layer - blue line and status values (e.g., alpha, gamma, and epsilon)
        :return:
        """
        # empty out all the blue lines before we start drawing more of them.
        self.things_to_draw = []

        self.draw_Q_values(batch, q_group)
        self.draw_last_path(batch, last_path_group)
        self.draw_current_path(batch, current_path_group)
        self.draw_stats(batch, last_path_group)

    def draw_Q_values(self, batch: pyglet.graphics.Batch, q_group: pyglet.graphics.Group) -> None:
        """
        loop through all the squares to draw the numbers for the Q values.
        :param batch: the batch into which we are drawing
        :param q_group: the graphics group (i.e., the layer) on which we are drawing this part.
        :return: None
        """
        for r in range(len(self.Q_grid)):
            for c in range(len(self.Q_grid[0])):
                # skip any cells that are zero - just leave them blank and move onto the next cell.
                if self.Q_grid[r][c] == 0:
                    continue
                # print the number of the cell's Q value.
                self.things_to_draw.append(pyglet.text.Label(text=f"{self.Q_grid[r][c]:.1f}",
                                                             font_name="Arial",
                                                             font_size=12,  # int(CellFile.CELL_SIZE/2),
                                                             color=(0, 0, 0, 255),
                                                             y=int(CellFile.CELL_SIZE * (len(self.Q_grid) - r - 0.5)),
                                                             x=int(CellFile.CELL_SIZE * (c + 0.5)),
                                                             anchor_x='center',
                                                             anchor_y='center',
                                                             batch=batch,
                                                             group=q_group))

    def draw_last_path(self, batch: pyglet.graphics.Batch, path_group: pyglet.graphics.Group) -> None:
        """
        Now draw the path taken by the latest robot... if there is a path to draw....
        :param batch:
        :param path_group:
        :return:
        """
        if len(self.latest_path) > 0:
            # loop through all the stops along the path.
            for i in range(1, len(self.latest_path)):

                # print(f"{x=}\t{y=}\t{x2=}\t{y2=}")
                self.things_to_draw.append(pyglet.shapes.Line(
                                     x=int((self.latest_path[i - 1][1] + 0.5) * CellFile.CELL_SIZE)-2,
                                     y=int((len(self.Q_grid) - self.latest_path[i - 1][0] - 0.5) *
                                           CellFile.CELL_SIZE) + 2,
                                     x2=int((self.latest_path[i][1] + 0.5) * CellFile.CELL_SIZE)-2,
                                     y2=int((len(self.Q_grid) - self.latest_path[i][0] - 0.5) *
                                            CellFile.CELL_SIZE) + 2,
                                     thickness=3,
                                     color=(0, 0, 255),  # blue
                                     batch=batch,
                                     group=path_group))

    def draw_current_path(self, batch: pyglet.graphics.Batch, path_group: pyglet.graphics.Group) -> None:
        if self.should_draw_robot:
            if len(self.current_path) > 0:
                # loop through all the stops along the path.
                for i in range(1, len(self.current_path)):
                    # print(f"{x=}\t{y=}\t{x2=}\t{y2=}")
                    self.things_to_draw.append(
                        pyglet.shapes.Line(x=int((self.current_path[i - 1][1] + 0.5) * CellFile.CELL_SIZE)+2,
                                           y=int((len(self.Q_grid) - self.current_path[i - 1][0] - 0.5) *
                                                 CellFile.CELL_SIZE)-2,
                                           x2=int((self.current_path[i][1] + 0.5) * CellFile.CELL_SIZE)+2,
                                           y2=int((len(self.Q_grid) - self.current_path[i][0] - 0.5) *
                                                  CellFile.CELL_SIZE) - 2,
                                           thickness=3,
                                           color=(64, 255, 255),
                                           batch=batch,
                                           group=path_group))
            current_dot = pyglet.shapes.Circle(x=int((self.current_coords[1] + 0.5) * CellFile.CELL_SIZE),
                                               y=int((len(self.Q_grid) - self.current_coords[0]-0.5) *
                                                     CellFile.CELL_SIZE),
                                               radius=int(CellFile.CELL_SIZE * 0.4),
                                               color=(64, 255, 255),
                                               batch=batch,
                                               group=path_group)
            current_dot.opacity = 128
            self.things_to_draw.append(current_dot)

    def draw_stats(self, batch: pyglet.graphics.Batch, path_group: pyglet.graphics.Group):
        # other labels to show at the top of the window.
        self.things_to_draw.append(pyglet.text.Label(text=f"𝛼 = {self.learning_rate:.2f}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid)+1)*CellFile.CELL_SIZE-2,
                                                     x=2,
                                                     anchor_x='left',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))
        self.things_to_draw.append(pyglet.text.Label(text=f"ɣ = {self.discount_factor:.2f}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid) + 1) * CellFile.CELL_SIZE - 2,
                                                     x=int(len(self.Q_grid[0])*CellFile.CELL_SIZE/2),
                                                     anchor_x='center',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))
        self.things_to_draw.append(pyglet.text.Label(text=f"ε = {self.exploration_ratio:.2f}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid) + 1) * CellFile.CELL_SIZE - 2,
                                                     x=int(len(self.Q_grid[0]) * CellFile.CELL_SIZE)-2,
                                                     anchor_x='right',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))
        self.things_to_draw.append(pyglet.text.Label(text=f"current R = {self.total_reward_for_current_run}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid) + 1) * CellFile.CELL_SIZE - 20,
                                                     x=2,
                                                     anchor_x='left',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))

        self.things_to_draw.append(pyglet.text.Label(text=f"last R = {self.latest_reward}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid) + 1) * CellFile.CELL_SIZE - 20,
                                                     x=int(len(self.Q_grid[0])*CellFile.CELL_SIZE/2),
                                                     anchor_x='center',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))

        self.things_to_draw.append(pyglet.text.Label(text=f"Run #: {self.run_num}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid) + 1) * CellFile.CELL_SIZE - 20,
                                                     x=int(len(self.Q_grid[0]) * CellFile.CELL_SIZE)-2,
                                                     anchor_x='right',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))

        self.things_to_draw.append(pyglet.text.Label(text=f"mean reward = {self.mean_reward:.1f}",
                                                     font_name="Arial",
                                                     font_size=STAT_FONT_SIZE,
                                                     color=(255, 255, 255, 255),
                                                     y=(len(self.Q_grid) + 1) * CellFile.CELL_SIZE - 40,
                                                     x=2,
                                                     anchor_x='left',
                                                     anchor_y='top',
                                                     batch=batch,
                                                     group=path_group))
