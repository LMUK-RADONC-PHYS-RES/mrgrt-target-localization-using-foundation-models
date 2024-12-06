   

import numpy as np

def generate_sequence(width, height, num_frames, radius, amplitude, frequency, dtype=np.float32):
    # This function generates an example of an 2d+t image sequence
    # by strategically morphing a single image.
    #       -       -       -       -
    # o---/   \   /   \   /   \   /   
    #           -       -       -   

    # Create a grid of coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy])

    # Create a sequence of frames
    sequence = np.zeros((num_frames, height, width), dtype=dtype)
    path = np.zeros((num_frames, 2), dtype=np.float32)
    for t in range(num_frames):
        # Compute the position of the circle
        center_x = 0
        center_y = amplitude * np.sin(2 * np.pi * frequency * t / num_frames)
        center = np.array([center_x, center_y])
        path[t] = center
        # Compute the distance to the circle
        distance = np.linalg.norm(grid - center[:, None, None], axis=0)

        # Create the frame
        frame = (distance < radius).astype(dtype)
        sequence[t] = frame

    return sequence, path