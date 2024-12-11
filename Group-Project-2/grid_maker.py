import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

"""BELOW YOU CAN ADJUST SOME GENERAL SETTINGS."""

HIGHWAY = 120
STREET = 15  # minimum speed is 5
ROAD = 30
MAIN_ROAD = 50
CITY_SPACING_FACTOR = 3.5  # should be minimal 2.3 for 4 or less cities and 3.5 for 5 cities
MAIN_ROAD_DISTANCE = 10  # This should be at least 8
HIGHWAY_SPACING = 10  # This should be at least 3
PATH_TRANSPARENCY = 1

""" DO NOT CHANGE THE CODE BELOW! """

RNG = np.random.default_rng()

def create_speed_cmap():
    low_speed = plt.get_cmap('YlGnBu', 256)
    high_speed = plt.get_cmap('brg', 256)
    low_speed_colors = low_speed(np.linspace(0, 1, 108))
    housing_color = low_speed_colors[0]
    low_speed_colors[32:] = low_speed(np.linspace(0, 1, 106))[10:-20]
    high_speed_colors = high_speed(np.linspace(0, 1, 256))
    path_color = np.array([0, 0, 0, 1])
    speed = np.vstack((low_speed_colors, high_speed_colors[38:186]))
    speed[-32:, :] = path_color
    speed[:32, :] = housing_color
    return colors.ListedColormap(speed)

class Map():
    def __init__(self, difficulty=0, side_roads=(2, 4), cities=2):
        out = DIFF_GRID[difficulty](side_roads, cities)
        if isinstance(out[1], list):
            self.grid, self.city_corners, self.city_grids = out
        else:
            self.grid = out
            self.city_corners = [(0, 0)]
            self.city_grids = [self.grid]

    def get_coordinate(self):
        """
        Get a random coordinate of the map.
        """
        return tuple(map(int, RNG.choice(list(zip(*np.where(self.grid > 1))))))

    def get_coordinate_in_city(self, city_n=0):
        """
        Gets a random coordinate in a specific city.
        """
        local_coor = tuple(RNG.choice(list(zip(*np.where(self.city_grids[city_n] > 1)))))
        return int(self.city_corners[city_n][0] + local_coor[0]), int(self.city_corners[city_n][1] + local_coor[1])

    def get_all_city_exits(self):
        """
        This gets all city exits of a map.
        """
        return list(zip(*np.where((self.get_highway_map().grid > 0) & (self.get_city_map().grid > 0))))

    def get_city_map(self):
        """
        This give back a copy of the map without highways, but with highway exits.
        """
        tmp = copy.deepcopy(self)
        tmp[self.grid == HIGHWAY] = 0
        highways = self.get_highway_map()
        for i in range(1, tmp.shape[0] - 1):
            for j in range(1, tmp.shape[1] - 1):
                if not highways[i, j]:
                    continue

                if np.sum(highways[i-1:i+2, j-1:j+2]) < HIGHWAY * 3:
                    tmp[i,j] = MAIN_ROAD
        return tmp

    def get_highway_map(self):
        """
        This give back a copy of the map with only highways.
        """
        tmp = copy.deepcopy(self)
        tmp[self.grid != HIGHWAY] = 0
        return tmp

    def __getitem__(self, item):
        """
        This makes the map indexable.
        """
        try:
            return self.grid[item]
        except IndexError as error:
            raise IndexError(f"A map object can only be index similar to a numpy array, the following error was cast by numpy:\n\t\t    {error}")

    def __setitem__(self, item, value):
        try:
            self.grid[item] = value
        except IndexError as error:
            raise IndexError(f"A map object can only be index similar to a numpy array, the following error was cast by numpy:\n\t\t    {error}")

    def __repr__(self):
        return repr(self.grid)

    def show(self, path=None, axis=False):
        """
        This shows the map, with or without a path and with or without axis (coordinate numbers).
        """
        path_grid = np.zeros(self.grid.shape)
        if path is not None:
            prev_node = path[0]
            for node in path[1:]:
                x1, x2 = min(prev_node[0], node[0]), max(prev_node[0], node[0])
                y1, y2 = min(prev_node[1], node[1]), max(prev_node[1], node[1])
                path_grid[x1:x2+1, y1:y2+1] = 1
                prev_node = node

        plt.imshow(self.grid, vmin=-(HIGHWAY+19)/6, vmax=(HIGHWAY+20)*7/6, cmap=SPEED_CMAP)
        speed_range = list(range(0, (HIGHWAY+20)*7//6, 20))
        cbar = plt.colorbar(ticks=[-(HIGHWAY+19)/6] + speed_range, fraction=0.045, pad=0.04)
        cbar.ax.set_yticklabels(["houses"] + speed_range[:-1] + ["path"])  # vertically oriented colorbar
        plt.imshow(path_grid, cmap="binary", alpha=path_grid*PATH_TRANSPARENCY)
        if not axis:
            plt.axis('off')
        plt.show()

    @property
    def shape(self):
        return self.grid.shape

    @property
    def size(self):
        return self.grid.size

    @staticmethod
    def create_small_grid(side_roads=(1, 3), *args, **kwargs):
        """
        This creates a small grid with a few side roads.
        The number of side roads is random.

        :param side_roads: The minimum and maximum number of side roads
        :type side_roads: tuple[int]
        """
        if not isinstance(side_roads, tuple):
            raise ValueError("side_roads should be a tuple with a min and max, e.g., (1, 3)!")

        return Map._small_grid(side_roads)[0]

    @staticmethod
    def create_small_imperfect_grid(side_roads=(1,3), *args, **kwargs):
        """
        This creates a small grid with a few side roads.
        The number of side roads is random.

        :param side_roads: The minimum and maximum number of side roads
        :type side_roads: tuple[int]
        """
        if not isinstance(side_roads, tuple):
            raise ValueError("side_roads should be a tuple with a min and max, e.g., (1, 3)!")

        if side_roads[0] == 0:
            raise ValueError("An imperfect grid needs at least one side road!")

        grid, rows, cols = Map._small_grid(side_roads)

        # remove row
        for row in RNG.choice(rows[1:-1], size=RNG.integers(1, len(rows)*len(cols)//4)):
            col_id = RNG.integers(1, len(cols))
            grid[row, cols[col_id-1]+1:cols[col_id]] = 0

        # remove col
        for col in RNG.choice(cols[1:-1], size=RNG.integers(1, len(rows)*len(cols)//4)):
            row_id = RNG.integers(1, len(rows))
            grid[rows[row_id-1]+1:rows[row_id], col] = 0

        # check unreachable coordinates
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                if not grid[i, j]:
                    continue

                # check for solo coordinate
                if not np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]:
                    grid[i, j] = 0

        return grid

    @staticmethod
    def create_medium_grid(side_roads=(1,3), *args, **kwargs):
        """
        This creates a medium grid with 4 smaller grids

        :param side_roads: The minimum and maximum number of side roads of each block
        :type side_roads: tuple[int]
        """
        if not isinstance(side_roads, tuple):
            raise ValueError("side_roads should be a tuple with a min and max, e.g., (1, 3)!")

        invalid_grid = True
        while invalid_grid:
            grid = Map._combine_grids(side_roads, Map.create_small_imperfect_grid)
            invalid_grid = not Map._check_grid(grid)

        return grid

    @staticmethod
    def create_large_grid(side_roads=(1,3), *args, **kwargs):
        """
        This creates a large grid with 4 medium grids

        :param side_roads: The minimum and maximum number of side roads of each block
        :type side_roads: tuple[int]
        """
        if not isinstance(side_roads, tuple):
            raise ValueError("side_roads should be a tuple with a min and max, e.g., (1, 3)!")

        invalid_grid = True
        while invalid_grid:
            grid = Map._combine_grids(side_roads, Map.create_medium_grid)
            invalid_grid = not Map._check_grid(grid)

        return grid

    @staticmethod
    def create_city(side_roads=(1,3), *args, **kwargs):
        """
        This creates a city grid consisting of 4 large grids

        :param side_roads: The minimum and maximum number of side roads of each block
        :type side_roads: tuple[int]
        """
        if not isinstance(side_roads, tuple):
            raise ValueError("side_roads should be a tuple with a min and max, e.g., (1, 3)!")

        invalid_grid = True
        while invalid_grid:
            grid = Map._combine_grids(side_roads, Map.create_large_grid)
            invalid_grid = not Map._check_grid(grid)

        return grid

    @staticmethod
    def create_country(side_roads=(1,3), cities=2, *args, **kwargs):
        """
        This creates a large grid with 4 medium grids

        :param cities: The number of cities in a country
        :type cities: int
        :param side_roads: The minimum and maximum number of side roads of each block
        :type side_roads: tuple[int]
        """
        if cities < 2 or cities > 5:
            raise ValueError("Only countries with 2,3,4 or 5 cities can be generated!")

        old_city_grids = [Map.create_city(side_roads) for _ in range(cities)]
        # add fake city (for spacing)
        if cities == 2:
            old_city_grids.append(np.zeros(old_city_grids[0].shape))

        # determine the dimensions of the grid
        max_city_width = max([city.shape[1] for city in old_city_grids])
        max_city_height = max([city.shape[0] for city in old_city_grids])

        country_grid = np.zeros((int(max_city_height * CITY_SPACING_FACTOR),
                                 int(max_city_width * CITY_SPACING_FACTOR)))

        Map._make_highways(country_grid, old_city_grids)
        city_grids, spacings = Map._make_ring_highways(old_city_grids)
        city_corners = [(spacings[0], spacings[0]),
                        (spacings[1], country_grid.shape[1] - city_grids[1].shape[1] + spacings[1])]

        # add grids to the new grid
        country_grid[:city_grids[0].shape[0], :city_grids[0].shape[1]] = city_grids[0]
        country_grid[:city_grids[1].shape[0], -city_grids[1].shape[1]:] = city_grids[1]
        if cities > 2:
            country_grid[-city_grids[2].shape[0]:, :city_grids[2].shape[1]] = city_grids[2]
            city_corners.append((country_grid.shape[0] - city_grids[2].shape[0] + spacings[2], spacings[2]))
        if cities > 3:
            country_grid[-city_grids[3].shape[0]:, -city_grids[3].shape[1]:] = city_grids[3]
            city_corners.append((country_grid.shape[0] - city_grids[3].shape[0] + spacings[3],
                                 country_grid.shape[1] - city_grids[3].shape[1] + spacings[3]))
        if cities > 4:
            country_grid[(country_grid.shape[0] - city_grids[4].shape[0]) // 2:
                         (country_grid.shape[0] + city_grids[4].shape[0]) // 2,
                         (country_grid.shape[1] - city_grids[4].shape[1]) // 2:
                         (country_grid.shape[1] + city_grids[4].shape[1]) // 2] = city_grids[4]
            city_corners.append(((country_grid.shape[0] - city_grids[4].shape[0]) // 2 + spacings[4],
                                 (country_grid.shape[1] - city_grids[4].shape[1]) // 2 + spacings[4]))
            
        return country_grid, city_corners, old_city_grids

    @staticmethod
    def _make_highways(country_grid, city_grids):
        """
        Default highway grid
        """
        outer_left = RNG.integers(city_grids[0].shape[1] // 2)
        outer_right = RNG.integers(country_grid.shape[1] - city_grids[1].shape[1] // 2, country_grid.shape[1])
        outer_top = RNG.integers(city_grids[0].shape[0] // 2 )
        outer_bottom = RNG.integers(country_grid.shape[0] - city_grids[2].shape[0] // 2, country_grid.shape[0])

        cols = [outer_left,
                outer_right,
                RNG.integers(outer_left + HIGHWAY_SPACING, city_grids[0].shape[1]),  # inner left
                RNG.integers(country_grid.shape[1] - city_grids[1].shape[1], outer_right - MAIN_ROAD_DISTANCE),  # inner right
                country_grid.shape[1] // 2 + RNG.integers(-HIGHWAY_SPACING, HIGHWAY_SPACING)]  # middle
        rows = [outer_top,
                outer_bottom,
                RNG.integers(outer_top + HIGHWAY_SPACING, city_grids[0].shape[0]),  # inner top
                RNG.integers(country_grid.shape[0] - city_grids[2].shape[0], outer_bottom - MAIN_ROAD_DISTANCE),  # inner bottom
                country_grid.shape[0] // 2 + RNG.integers(-HIGHWAY_SPACING, HIGHWAY_SPACING)]  # middle

        for row in rows:
            country_grid[row, outer_left:outer_right+1] = HIGHWAY
        for col in cols:
            country_grid[outer_top:outer_bottom+1, col] = HIGHWAY

    @staticmethod
    def _make_ring_highways(city_grids):
        """
        Make the ring highways and exits
        """
        cities = []
        spacings = []
        for city_grid in city_grids:
            spacing = RNG.integers(3, HIGHWAY_SPACING)
            new_city = np.ones((city_grid.shape[0] + spacing*2, city_grid.shape[1] + spacing*2)) * HIGHWAY
            new_city[1:-1, 1:-1] = 0

            # Make exits rows
            side = RNG.choice([0,1])
            for exit_ in sorted(RNG.integers(spacing, city_grid.shape[0] + spacing, size=RNG.integers(1, 9))):
                if side:
                    new_city[exit_, :HIGHWAY_SPACING] = HIGHWAY
                else:
                    new_city[exit_, -HIGHWAY_SPACING:] = HIGHWAY
                side = 1- side

            # Make exits cols
            side = RNG.choice([0,1])
            for exit_ in sorted(RNG.integers(spacing, city_grid.shape[1] + spacing, size=RNG.integers(1, 9))):
                if side:
                    new_city[:HIGHWAY_SPACING, exit_] = HIGHWAY
                else:
                    new_city[-HIGHWAY_SPACING:, exit_] = HIGHWAY
                side = 1 - side

            new_city[spacing:-spacing, spacing:-spacing] = city_grid
            cities.append(new_city)
            spacings.append(spacing)
        return cities, spacings

    @staticmethod
    def _check_grid(grid):
        # check grid
        for i in range(grid.shape[0]-1):
            for j in range(grid.shape[1]-1):
                if 0 not in grid[i:i+2, j:j+2]:
                    return False
        return True

    @staticmethod
    def _small_grid(side_roads):
        side_roads = side_roads[0] + 1, side_roads[1] + 2

        rows = [0] + list(np.cumsum(RNG.integers(2, 5, size=RNG.integers(*side_roads))))
        cols = [0] + list(np.cumsum(RNG.integers(2, 5, size=RNG.integers(*side_roads))))
        grid = np.zeros((rows[-1]+1, cols[-1]+1))
        grid[rows] = STREET
        grid[:, cols] = STREET
        grid[0], grid[-1], grid[:,0], grid[:,-1] = ROAD, ROAD, ROAD, ROAD
        return grid, rows, cols

    @staticmethod
    def _combine_grids(side_roads, grid_func):
        """
        Combine small grids to form a larger grid
        """
        grids = [grid_func(side_roads) for _ in range(4)]
        rows = max(grids[0].shape[0] + grids[1].shape[0], grids[2].shape[0] + grids[3].shape[0])
        cols = max(grids[0].shape[1] + grids[2].shape[1], grids[1].shape[1] + grids[3].shape[1])

        large_grid = np.zeros((rows+1, cols+1))

        # add main roads
        main_road_h = {0,
                       grids[0].shape[0]-1,
                       grids[2].shape[0]-1,
                       large_grid.shape[0] - grids[1].shape[0],
                       large_grid.shape[0] - grids[3].shape[0],
                       large_grid.shape[0]-1}
        prev_row = -MAIN_ROAD_DISTANCE
        for row in sorted(main_road_h):
            # add main cross roads
            if abs(prev_row - row) >= 3:
                large_grid[row] = ROAD if grid_func == Map.create_small_imperfect_grid else MAIN_ROAD
            # At extra cross roads for large grids
            new_row = prev_row+MAIN_ROAD_DISTANCE
            while new_row < row-MAIN_ROAD_DISTANCE:
                large_grid[new_row] = ROAD if grid_func == Map.create_small_imperfect_grid else MAIN_ROAD
                new_row += RNG.integers(MAIN_ROAD_DISTANCE-5,MAIN_ROAD_DISTANCE+6)

            prev_row = row

        # add main roads
        main_road_v = {0,
                       grids[0].shape[1]-1,
                       grids[1].shape[1]-1,
                       large_grid.shape[1] - grids[2].shape[1],
                       large_grid.shape[1] - grids[3].shape[1],
                       large_grid.shape[1] - 1}
        prev_col = -MAIN_ROAD_DISTANCE
        for col in sorted(main_road_v):
            # add main cross roads
            if abs(prev_col - col) >= 3:
                large_grid[:, col] = ROAD if grid_func == Map.create_small_imperfect_grid else MAIN_ROAD
            # At extra cross roads for large grids
            new_col = prev_col+MAIN_ROAD_DISTANCE
            while new_col < col-MAIN_ROAD_DISTANCE:
                large_grid[:, new_col] = ROAD if grid_func == Map.create_small_imperfect_grid else MAIN_ROAD
                new_col += RNG.integers(MAIN_ROAD_DISTANCE-5,MAIN_ROAD_DISTANCE+6)
            prev_col = col

        # upper left
        large_grid[:grids[0].shape[0], :grids[0].shape[1]] = grids[0]
        # lower left
        large_grid[-grids[1].shape[0]:, :grids[1].shape[1]] = grids[1]
        # upper right
        large_grid[:grids[2].shape[0], -grids[2].shape[1]:] = grids[2]
        # lower left
        large_grid[-grids[3].shape[0]:, -grids[3].shape[1]:] = grids[3]

        # set the boundary
        large_grid[0], large_grid[-1], large_grid[:,0], large_grid[:,-1] = MAIN_ROAD, MAIN_ROAD, MAIN_ROAD, MAIN_ROAD

        return large_grid


SPEED_CMAP = create_speed_cmap()
DIFF_GRID = {
    0: Map.create_small_grid,
    1: Map.create_small_imperfect_grid,
    2: Map.create_medium_grid,
    3: Map.create_large_grid,
    4: Map.create_city,
    5: Map.create_country
}