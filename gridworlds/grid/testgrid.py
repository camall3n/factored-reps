from . import basicgrid

class TestGrid(basicgrid.BasicGrid):
    def __init__(self):
        super().__init__(rows=3, cols=4)
        self._grid[1,4] = 1
        self._grid[2,3] = 1
        self._grid[3,2] = 1
        self._grid[5,4] = 1
        self._grid[4,7] = 1

# Should look roughly like this:
# _______
#|  _|   |
#| |    _|
#|___|___|
