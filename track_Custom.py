import numpy as np
import matplotlib.pyplot as plt

class Track:
    def __init__(self, track_length=100, uniform_width=5):
        self.track_length = track_length
        self.width = uniform_width
        
        # Set the two curvatures: curvature_1 and curvature_2
        self.curvatures = self._generate_smooth_transition_curvatures()

    def get_width(self):
        return self.width
    
    def _generate_smooth_transition_curvatures(self):
        curvature_1 = np.random.uniform(0.05, 0.2)  # Curvature for the first segment
        curvature_2 = np.random.uniform(-0.05, -0.2)  # Curvature for the second segment

        # Initializing curvatures list
        curvatures = []
        segment_length = self.track_length // 2

        # First segment: curvature_1
        for i in range(segment_length):
            curvatures.append(curvature_1)

        # Second segment: curvature_2, starting directly after curvature_1
        for i in range(segment_length, self.track_length):
            curvatures.append(curvature_2)

        return curvatures

    def get_curvature(self, x_position):
        # Ensure x_position is within the bounds of the curvatures list
        index = min(int(x_position), self.track_length - 1)
        return self.curvatures[index]

    def get_left_boundary(self, x_position):
        # Adjust x_position by subtracting 50 if x_position is greater than or equal to 50
        new_x = x_position
        if x_position >= 50:
            new_x -= 50
        
        # Calculate the left boundary based on the curvature at the adjusted x_position
        curvature = self.get_curvature(x_position)
        left_boundary = -self.width / 2 + curvature * new_x
        
        # Calculate the dynamic offset for the first 50 units based on curvature
        if x_position >= 50:
            offset = sum([self.get_curvature(x) * 1 for x in range(50)])  # Sum of curvature changes
            left_boundary += offset
            
        return left_boundary

    def get_right_boundary(self, x_position):
        # Adjust x_position by subtracting 50 if x_position is greater than or equal to 50
        new_x = x_position
        if x_position >= 50:
            new_x -= 50
        
        # Calculate the right boundary based on the curvature at the adjusted x_position
        curvature = self.get_curvature(x_position)
        right_boundary = self.width / 2 + curvature * new_x
        
        # Calculate the dynamic offset for the first 50 units based on curvature
        if x_position >= 50:
            offset = sum([self.get_curvature(x) * 1 for x in range(50)])  # Sum of curvature changes
            right_boundary += offset
            
        return right_boundary

# # Instantiate the track object
# track = Track(track_length=100, uniform_width=10)

# # Generate x_positions for smooth plotting
# x_positions = np.linspace(0, track.track_length, 1000)

# # Calculate the left and right boundaries
# left_boundaries = [track.get_left_boundary(x) for x in x_positions]
# right_boundaries = [track.get_right_boundary(x) for x in x_positions]

# # Middle curve (track centerline) is the average of the left and right boundaries
# middle_curve = [(left + right) / 2 for left, right in zip(left_boundaries, right_boundaries)]

# # Plot the track boundaries with the generated path and middle curve
# plt.figure(figsize=(10, 6))
# plt.plot(x_positions, middle_curve, label="Middle Curve", color='blue', linestyle='--')
# plt.plot(x_positions, left_boundaries, label="Left Boundary", color='red')
# plt.plot(x_positions, right_boundaries, label="Right Boundary", color='green')
# plt.fill_between(x_positions, left_boundaries, right_boundaries, color='gray', alpha=0.5)
# plt.xlabel('Track Position (x)')
# plt.ylabel('Track Elevation (y)')
# plt.title('Track Boundaries and Middle Curve Based on Curvature')
# plt.legend()
# plt.grid(True)
# plt.show()
