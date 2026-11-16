import logging
from typing import Optional, Tuple, List

import LearnerFile
from MapFile import Map
from TransitionManagerFile import TransitionManager
from LearnerFile import Learner
from WindowFile import Window
import pyglet
import csv

# LOGGING SETUP ===================================================================================================
logging.basicConfig(level=logging.INFO)  # simple version to the output console
# logging.basicConfig(level=logging.DEBUG, filename=f"log {datetime.datetime.now():%m-%d@%H:%M:%S}.txt",
#                     format="%(asctime)s %(levelname)s %(message)s",
#                     datefmt="%H:%M:%S %p --- ")  # more robust, sent to a file cNode = Tuple[int, T]

# STARTING CLASS ==================================================================================================


class QLearnRunner:
    def __init__(self, map_filename: str):
        self.__delay_time = 1.0
        self.__step_mode: int = LearnerFile.STEP_MODE_SINGLE_STEP

        self.__window: Optional[Window] = None
        self.__main_map: Optional[Map] = None
        self.__start_position: Tuple[int, int] = (-1, -1)
        self.__is_animating = True

        self.load_data(map_filename)
        self.__Q_grid: List[List[float]] = [[0.0 for _c in range(self.__main_map.shape()[1])]
                                             for _r in range(self.__main_map.shape()[0])]
        self.__transition_manager = TransitionManager(self.__main_map, self.__Q_grid)
        self.__learner = Learner(self.__Q_grid, self.__transition_manager, self.__start_position)

    def start(self, step_mode:int = LearnerFile.STEP_MODE_SINGLE_STEP, time_step: float = 1.0):
        # This is what calls "perform_learning_iteration()" periodically. The number here is the seconds between calls.
        self.__delay_time = time_step
        self.__step_mode = step_mode
        pyglet.clock.schedule_interval(self.perform_learning_iteration, self.__delay_time)

        # set up the window that will show the graphics
        self.__window = Window(self.__main_map, self.__learner, self.toggle_animation)
        pyglet.app.run()

    def load_data(self, filename: str):
        """
        Loads the map from a tab-separated-values (tsv) file, as well as the starting location on the map
        :return:
        """
        tsv_file = open(filename)
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        line_num = 0
        r = 0  # start at top row.
        num_rows = 0
        num_cols = 0
        for line in read_tsv:  # read one line of the file at a time. Keep going while there still are lines.

            if line_num == 0:  # first line of file determines the size of the map
                num_rows, num_cols = self.parse_map_size(line)
                self.__main_map = Map(num_rows, num_cols)

            elif line_num == 1:  # second line of the file determines the starting location
                self.parse_start_position(line)
            else:  # all other lines of file are the rewards for the various cells.
                c = 0
                for item in line:
                    if item == "":  # empty spots treated like zero
                        self.__main_map.get_cell_at(r, c).reward = 0
                    else:
                        self.__main_map.get_cell_at(r, c).reward = int(item)
                    c += 1
                    if c >= num_cols:  # safety check that we don't go past the horizontal size of the grid
                        break
                r += 1
                if r >= num_rows:  # safety check that we don't go past the vertical size of the grid
                    break
            line_num += 1
            if num_rows <= 0 or num_cols <= 0:
                raise Exception("Something went wrong reading the number of rows and columns in this file.")
        tsv_file.close()
        logging.info("Map file is loaded")

    @staticmethod
    def parse_map_size(line: list[str]) -> tuple[int, int]:
        num_cols = int(line[0])
        num_rows = int(line[1])
        logging.info(f"Map size: ({num_rows}, {num_cols})")
        return num_cols, num_rows

    def parse_start_position(self, line: list[str]):
        self.__start_position = (int(line[0]), int(line[1]))
        logging.info(f"Start Position: {self.__start_position}")
        self.__main_map.set_start(self.__start_position)

    def toggle_animation(self):
        """
        starts/stops whether the animation is running, cancelling or reactivating the scheduled calls to
        self.perform_learning_iteration() and updating self.__is_animating to reflect the new status.
        :return: None
        """
        if self.__is_animating:
            pyglet.clock.unschedule(self.perform_learning_iteration)
        else:
            pyglet.clock.schedule_interval(self.perform_learning_iteration, self.__delay_time)
        self.__is_animating = not self.__is_animating

    def perform_learning_iteration(self, _dt):  # the underscore marks the dt parameter as unused.
        """
        tells the learner to attempt a path, or one step along a path, depending on the value of "step_mode."
        Then it updates what should be shown in the window... the next time window refreshes.
        :param _dt: the amount of time since the last time we called this. (Unused, but required)
        :return: None
        """
        if self.__step_mode == LearnerFile.STEP_MODE_ONE_PATH:
            self.__learner.explore_path()
        elif self.__step_mode == LearnerFile.STEP_MODE_SINGLE_STEP:
            continue_on_path = self.__learner.do_step()
            if not continue_on_path:
                self.__learner.reset_robot()
        else:
            raise Exception("self.__step_mode is not an allowed value.")

        if self.__window is not None:
            self.__window.update()


if __name__ == "__main__":
    QLRunner = QLearnRunner(map_filename="MapExample1.tsv")
    QLRunner.start(step_mode=LearnerFile.STEP_MODE_SINGLE_STEP, time_step=0.25)
    # QLRunner.start(step_mode=LearnerFile.STEP_MODE_ONE_PATH, time_step=0.05)
