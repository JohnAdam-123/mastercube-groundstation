import os
import tensorflow as tf
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset

def waymo_tfrecord_to_npy(tfrecord_path, output_folder, max_frames=None):
    """
    Convert Waymo .tfrecord files into .npy frames.
    
    Args:
        tfrecord_path (str): Path to the Waymo .tfrecord file.
        output_folder (str): Directory to save the .npy frames.
        max_frames (int, optional): Maximum number of frames to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    for i, data in enumerate(dataset):
        if max_frames is not None and i >= max_frames:
            break

        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())
        
        # Example: save front camera image only
        front_image = frame.images[0].image  # first camera
        image_array = np.frombuffer(front_image, dtype=np.uint8)
        
        npy_filename = os.path.join(output_folder, f"frame_{i:05d}.npy")
        np.save(npy_filename, image_array)
        
        if i % 50 == 0:
            print(f"Saved frame {i}")

    print(f"Conversion done! {i+1} frames saved in {output_folder}")


    if __name__ == "__main__":
        tfrecord_file = "./datasets/waymo/train.tfrecord"  # Path to your Waymo file
        output_folder = "./datasets/waymo_npy"
        waymo_tfrecord_to_npy(tfrecord_file, output_folder, max_frames=100)