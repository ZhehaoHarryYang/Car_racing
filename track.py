# track.py
TRACK_WIDTH = 5  # Width of the track
STRAIGHT_LENGTH = 50  # Length of the straight section
SLOPE = 0.5  # Slope of the turn (horizontal 1, vertical 0.3)
TRACK_LENGTH = 100  # Total length of the track

def track_center_y(x_position):
    """Calculate the y position of the center of the track for a given x position."""
    if x_position <= STRAIGHT_LENGTH:
        return 0  # Center y-position is flat for the straight section
    else:
        # For the sloped section, increase y based on slope
        return SLOPE * (x_position - STRAIGHT_LENGTH)

def left_boundary(x_position):
    """Calculate the left boundary of the track at a given x position."""
    return track_center_y(x_position) - TRACK_WIDTH / 2

def right_boundary(x_position):
    """Calculate the right boundary of the track at a given x position."""
    return track_center_y(x_position) + TRACK_WIDTH / 2
