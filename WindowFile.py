from typing import Callable

import pyglet
from MapFile import Map
from LearnerFile import Learner
import CellFile


class Window (pyglet.window.Window):
    """
    This class (which descends from a generic Window class) describes the details about a window used in this program.
    """
    def __init__(self, rewards_map: Map, learner: Learner, toggle_animation: Callable):  # "Callable" means a method
        """
        start the window off with the given information and any variables we'll be using.
        :param rewards_map:
        :param learner:
        """
        # first call the "superclass" - this is what actually makes the window.
        super(Window, self).__init__(rewards_map.num_cols * CellFile.CELL_SIZE,
                                     (rewards_map.num_rows + 1) * CellFile.CELL_SIZE,
                                     "World")

        # copy the given information into longer-term variables.
        self.map = rewards_map
        self.learner = learner
        # these are variables we'll need when drawing the window. For the moment, we're setting them to None, so
        # the computer knows that it has to keep track of the variable, but it isn't holding anything yet.
        self.batch = None
        self.map_group = None
        self.Q_group = None
        self.last_path_group = None
        self.current_path_group = None
        self.toggle_animation_callback = toggle_animation
        # let the window know what it will need to draw.
        self.update()

    def on_draw(self):
        """
        This method gets called _often_ (as in fractions of a second) it is what makes the content of the window appear.
        :return:
        """
        self.clear()  # wipe the screen
        self.batch.draw()  # redraw the current contents.

    def on_mouse_release(self, x, y, button, modifiers):
        self.toggle_animation_callback()

    def update(self):
        """
        change what is being drawn each time in self.on_draw()
        :return:
        """
        # The "batch" is the list of all the drawing commands that will need to be done.
        self.batch = pyglet.graphics.Batch()

        # Ordered groups are like layers... all the stuff in ordered group 0 will be drawn first, then
        # ordered group 1 will get drawn on top of that, and so forth.
        self.map_group = pyglet.graphics.Group(order=0)

        self.map.draw_self(self.batch, self.map_group)

        if self.learner is not None:
            self.Q_group = pyglet.graphics.Group(order=1)
            self.last_path_group = pyglet.graphics.Group(order=2)
            self.current_path_group = pyglet.graphics.Group(order=3)
            self.learner.draw_self(self.batch, self.Q_group, self.last_path_group, self.current_path_group)
