import os
import tensorflow as tf
import numpy as np

def waymo_tfrecord_to_npy(tfrecord_path, output_folder, max_frames=None):
    """
    Converts a Waymo .tfrecord file to individual .npy frames.
    Each frame is saved as frame_00000.npy, frame_00001.npy, ...
    """
    os.makedirs(output_folder, exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    for i, data in enumerate(dataset):
        if max_frames is not None and i >= max_frames:
            break
        # Parse raw bytes into numpy array (simplified; you can expand parsing here)
        points = np.frombuffer(data.numpy(), dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
        np.save(os.path.join(output_folder, f"frame_{i:05d}.npy"), points)
    print(f"Saved {i+1} frames to {output_folder}")

if __name__ == "__main__":
    tfrecord_file = "/path/to/your/waymo/file.tfrecord"
    output_folder = "./datasets/waymo_npy"
    waymo_tfrecord_to_npy(tfrecord_file, output_folder, max_frames=100)