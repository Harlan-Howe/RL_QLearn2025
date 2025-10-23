import pyglet
from pyglet import shapes
from typing import List

CELL_SIZE = 60

class Cell:
    """
    a class representing a grid location in our world, including its location, visibility, and reward
    """

    def __init__(self, x:int, y:int, r: int, window_height:int):
        self.x = x
        self.y = y
        self.reward = r
        self.visited = False
        self.window_height = window_height


    def draw_self(self, drawn_shapes:List, batch: pyglet.graphics.Batch, layer: pyglet.graphics.Group, color_override = None):
        """
        gives the batch the instructions for drawing the given cell.
        :param drawn_shapes: the list of cells' drawing commands, so they are remembered
        :param batch: the list of commands to draw the next time we refresh the screen (which is often)
        :param layer: the first layer - the items drawn in this layer get drawn before other layers.
        :param color_override: by default, this is no color at all, but if a color is passed along, then the cell
                               will be this color despite all other status values. Primarily used to set the start.
        :return:
        """
        # picks a color to fill this cell in
        if color_override is not None:
            fill = color_override # if color_override is not None, just use that color and ignore all else.
        elif self.visited == False:
            fill = (128,128,128) # if the cell is not yet visited, set it to grey.
        elif self.reward == 0:
                fill = (255,255,255) # white - normal space
        elif self.reward > 0:
                fill = (64,64,255) # blue - reward!
        elif self.reward < -50 and self.reward > -9000:
                fill = (255,0,0) # red - penalty
        elif self.reward < 0 and self.reward >= -50:
                fill = (255,228+2*self.reward,228+2*self.reward) #pink - non-stopping penalty
        else:
            fill = (0, 0, 0)  # black (wall) if reward <= -9000

        # then draw this rectangle
        drawn_shapes.append(shapes.BorderedRectangle(x=CELL_SIZE * self.x,
                                             y=self.window_height - CELL_SIZE * self.y - CELL_SIZE,
                                             width=CELL_SIZE,
                                             height=CELL_SIZE,
                                             border = 1,
                                             color = fill, # fill color
                                             border_color = (64,64,64), # outline (a.k.a. "stroke") color
                                             batch=batch,
                                             group=layer
                                             ))

        if self.visited and (self.reward > 0 or self.reward < -50):
            drawn_shapes.append(shapes.BorderedRectangle(x = CELL_SIZE * self.x+4,
                                                         y = self.window_height - CELL_SIZE * self.y - CELL_SIZE +4,
                                                         width = CELL_SIZE - 8,
                                                         height= CELL_SIZE - 8,
                                                         border = 1,
                                                         color = fill,
                                                         border_color = (0,0,0),
                                                         batch=batch,
                                                         group = layer
                                                         ))
        if self.visited and (color_override == (255,255,0)):
            drawn_shapes.append(shapes.Circle(x = CELL_SIZE * (self.x+0.5) ,
                                              y = self.window_height - CELL_SIZE * (self.y-0.5) - CELL_SIZE,
                                              radius = CELL_SIZE/2 - 3,
                                              color=(0,0,0),
                                              batch=batch,
                                              group = layer
                                              ))
            drawn_shapes.append(shapes.Circle(x=CELL_SIZE * (self.x+0.5),
                                              y=self.window_height - CELL_SIZE * (self.y-0.5) - CELL_SIZE,
                                              radius=CELL_SIZE/2 - 4,
                                              color=fill,
                                              batch=batch,
                                              group=layer
                                              ))
