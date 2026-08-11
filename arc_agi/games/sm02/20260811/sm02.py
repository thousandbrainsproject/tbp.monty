from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    InteractionMode,
    Level,
    Sprite,
)

BACKGROUND_COLOR = 10

PADDING_COLOR = 4

# Create sprites dictionary with all sprite definitions
sprites = {
    "player": Sprite(
        pixels=[[2]],  # Mouse
        name="player",
        blocking=BlockingMode.BOUNDING_BOX,
        interaction=InteractionMode.TANGIBLE,
    ),
    "exit": Sprite(
        pixels=[[11]],  # Cheese
        name="exit",
        blocking=BlockingMode.BOUNDING_BOX,
        interaction=InteractionMode.TANGIBLE,
    ),
    "maze_1": Sprite(
        pixels=[
            [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],  # Row 0
            [13, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, 13],  # Row 1
            [13, -1, 13, -1, 13, -1, 13, 13, 13, 13, -1, 13],  # Row 2
            [13, -1, 13, -1, -1, -1, 13, -1, -1, -1, -1, 13],  # Row 3
            [13, -1, 13, 13,  0,  1,  2,  3, -1, 13, 13, 13],  # Row 4
            [13, -1, -1, -1,  4,  5,  6,  7, -1, -1, -1, 13],  # Row13
            [13, 13, 13, -1,  8,  9, 10, 11, 13, 13, -1, 13],  # Row 6
            [13, -1, -1, -1, 12, 13, 14, 15, -1, 13, -1, 13],  # Row 7
            [13, -1, 13, -1, 13, -1, -1, -1, -1, 13, -1, 13],  # Row 8
            [13, -1, 13, -1, 13, 13, -1, 13, -1, 13, -1, 13],  # Row 9
            [13, -1, 13, -1, -1, -1, -1, 13, -1, -1, -1, 13],  # Row 10
            [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],  # Row 11
        ],
        name="maze_1",
        blocking=BlockingMode.PIXEL_PERFECT,
        interaction=InteractionMode.TANGIBLE,
        layer=-1,  # Render below player and exit
    ),
    "maze_2": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # Row 0
            [5, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5],  # Row 1
            [5, -1, 5, -1, 5, -1, 5, 5, 5, 5, -1, 5],  # Row 2
            [5, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, 5],  # Row 3
            [5, -1, 5, 5, 5, 5, 5, 5, -1, 5, -1, 5],  # Row 4
            [5, -1, -1, -1, -1, -1, -1, 5, -1, 5, -1, 5],  # Row 5
            [5, 5, 5, 5, 5, 5, -1, 5, -1, 5, -1, 5],  # Row 6
            [5, -1, -1, -1, -1, 5, -1, 5, -1, 5, -1, 5],  # Row 7
            [5, -1, 5, 5, -1, 5, -1, 5, -1, 5, -1, 5],  # Row 8
            [5, -1, 5, -1, -1, 5, -1, -1, -1, 5, -1, 5],  # Row 9
            [5, -1, -1, -1, 5, 5, 5, 5, 5, 5, -1, 5],  # Row 10
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # Row 11
        ],
        name="maze_2",
        blocking=BlockingMode.PIXEL_PERFECT,
        interaction=InteractionMode.TANGIBLE,
        layer=-1,  # Render below player and exit
    ),
}

# Create levels array with all level definitions
levels = [
    # Level 1
    Level(
        sprites=[
            sprites["maze_1"].clone(),
            sprites["player"].clone().set_position(1, 10),  # Start position
            sprites["exit"].clone().set_position(10, 1),  # Exit position
        ],
        grid_size=(12, 12),
    ),
    # Level 2
    Level(
        sprites=[
            sprites["maze_2"].clone(),
            sprites["player"].clone().set_position(1, 1),  # Start position
            sprites["exit"].clone().set_position(10, 10),  # Exit position
        ],
        grid_size=(12, 12),
    ),
]


class Sm02(ARCBaseGame):
    """A simple maze game where the player navigates to the exit."""

    def __init__(self) -> None:
        """Initialize the SimpleMaze game."""
        # Create camera with background and padding colors
        camera = Camera(background=BACKGROUND_COLOR, letter_box=PADDING_COLOR)

        # Initialize the base game
        super().__init__(game_id="simple_maze", levels=levels, camera=camera)

    def step(self) -> None:
        """Step the game forward based on the current action."""
        # Handle movement based on action ID
        dx = 0
        dy = 0
        if self.action.id == GameAction.ACTION1:  # Move Up
            dy = -1
        elif self.action.id == GameAction.ACTION2:  # Move Down
            dy = 1
        elif self.action.id == GameAction.ACTION3:  # Move Left
            dx = -1
        elif self.action.id == GameAction.ACTION4:  # Move Right
            dx = 1

        collided = self.try_move("player", dx, dy)

        # Check if player collided with exit
        if collided and any(sprite.name == "exit" for sprite in collided):
            if self.is_last_level():
                # All levels completed, set game state to WIN
                self.win()
            else:
                self.next_level()

        self.complete_action()
