import tensorflow as tf
import numpy as np

from waymo_open_dataset import dataset_pb2 as open_dataset


def load_waymo_frame(file_path):
    """
    Load a single Waymo LiDAR frame saved as .npy.
    Returns a NumPy array of shape (N, 3) or (N, 4) if intensity is included.
    """
    try:
        data = np.load(file_path)
        if data.shape[1] > 4:
            data = data[:, :4]  # Keep XYZ + intensity
        return data
    except Exception as e:
        print(f"Error loading Waymo frame: {e}")
        return np.empty((0,3))

def load_waymo_points(tfrecord_path, output_dir, max_frames=None):
    """
    Load LiDAR points from Waymo TFRecord file.

    Args:
        tfrecord_path (str): Path to a Waymo .tfrecord file.
        output_dir (str): Directory to save the loaded frames.
        max_frames (int): Maximum number of frames to load - optional (for testing).

    Returns:
    points_list (list of np.ndarray): Each element is (N x 3) array of XYZ points for a frame.
    """
    points_list = []

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type = '')
    for i, data in enumerate(dataset):
        if max_frames is not None and i >= max_frames:
            break

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Combine all LiDAR lasers for this frame
        frame_points = []
        for lidar in frame.lasers:
            # Each laser has a list of returns (points)
            for point in lidar.ri_return1.range_image_return.range_image.pixel_return: # Optional: This can be simplified.
                x = point.x
                y = point.y
                z = point.z
                frame_points.append([x, y, z])

            # Appending all the lidar points as (N, 3)
            frame_points = extract_points_from_frame(frame) # This helper has been defined below.
            points_list.append(frame_points)

            if points_list:
                points = np.vstack(points_list) # (N, 3) for all points in this frame
                np.save(f"{output_dir}/waymo_frame_{i:04d}.npy", points) # Save as .npy for later use
                print(f"Saved frame {i} with shape {points.shape}")

        print("Conversion complete. Total frames loaded:", len(points_list))

def extract_points_from_frame(frame):
    """
    Extracts XYZ points from waymo frame (all LiDARs combined).

    Returns:
        np.array of shape (num_points, 3) containing XYZ coordinates.
    """
    import math
    all_points = []

    for laser in frame.lasers:
        # Use ri_return1 if multiple returns are present, otherwise use ri_return2
        range_image = laser.ri_return1.range_image
        if range_image is None:
            continue

        for r in range_image.row_return:
            for p in r.pixel_return:
                # Convert polar coordinates to Cartresian XYZ
                distance = p.range # In meters
                azimuth = p.azimuth # In radians
                elevation = p.elevation # In radians

                x = distance * math.cos(elevation) * math.sin(azimuth)
                y = distance * math.cos(elevation) * math.cos(azimuth)
                z = distance * math.sin(elevation)

                all_points.append([x, y, z])

    return np.array(all_points, dtype=np.float32)

