import CellFile
from CellFile import Cell
import pyglet
from typing import Tuple, List, Optional


class Map:
    """
    This class is what holds the information about the rewards for each cell in our world, as well as the
    starting location for all explorations.
    """
    def __init__(self, num_rows: int, num_cols: int):
        """
        Creates a Map with empty cells of the given size.
        :param num_cols:
        :param num_rows:
        """
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.start_pos = (-1, -1)  # temporary - we'll update this later.

        self.cell_grid: List[List[Optional[Cell]]] = [[None for _c in range(num_cols)]
                                                      for _r in range(num_rows)]
        for col in range(num_cols):
            for row in range(num_rows):
                # start each cell with reward = 0, and tell each cell where it is and how tall the grid
                # portion of the window is.
                self.cell_grid[row][col] = Cell(col, row, 0, self.num_rows * CellFile.CELL_SIZE)

        self.objects_to_draw = []

    def shape(self) -> Tuple[int, int]:
        """
        gets the number of rows and columns in this map.
        :return:
        """
        return self.num_rows, self.num_cols

    def set_start(self, p: Tuple[int, int]):
        """
        updates the starting location of all paths and sets this cell to be visited, so it will show up as yellow.
        :param p: (row, col) of the starting location
        :return:
        """
        self.start_pos = p
        self.get_cell_at(p[0], p[1]).visited = True

    def draw_self(self, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group):
        """
        tells the pygame graphics batch about drawing the cells in this map (note only the visited ones will be
        visible.)
        :param batch:  The batch job that will do the drawing when the window updates
        :param group: the layer within the window where these objects will be drawn.
        :return: None
        """
        # to make this work, we need to store the objects in memory, or they are forgotten before the batch draws them.
        # We start off by clearing all the old shapes from the last time we drew the map.
        self.objects_to_draw.clear()

        # loop through all the cells and tell each one to draw itself.
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                if row == self.start_pos[0] and col == self.start_pos[1]:  # if this is start, draw in yellow.
                    self.cell_grid[row][col].draw_self(self.objects_to_draw, batch, group, (255, 255, 0))
                else:
                    self.cell_grid[row][col].draw_self(self.objects_to_draw, batch, group)

    def get_cell_at(self, row: int, col: int) -> Cell:
        """
        an accessor for getting the cell variable at a given location. Mostly used by other classes.
        :param row:
        :param col:
        :return: the cell at this location.
        """
        return self.cell_grid[row][col]

    def __repr__(self):
        """
        If anybody tries to print the map, this is what it does to do that. (Not used in regular program, but it is
        helpful while debugging.)
        :return: a string describing this map.
        """
        output = ""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                output += f"{self.cell_grid[r][c].reward}\t"
            output += "\n"
        return output
