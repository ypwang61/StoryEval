import os
import cv2
import shutil
import math
import numpy as np
from decord import VideoReader, cpu
import base64
import re

clip_video_dir = "./_clips/" # directory to store the extracted frames



# Function to get the filename from a sentence
# the filename of prompt 'xxx' should be sentence_to_filename('xxx') + '.mp4'
def sentence_to_filename(sentence, max_length=198):
    # remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # split into words
    words = sentence.split()
    # concatenate with underscores
    filename = '_'.join(words)
    # truncate to max_length
    filename = filename[:max_length]

    return filename


# unified function to get the output file path
def get_output_file_path(result_dir, generative_model_name, eval_model_type, output_postfix):
    return os.path.join(
        result_dir, f"{generative_model_name}_{eval_model_type}{output_postfix}.json"
    )


def get_video_id(video_path):
    base_name = os.path.basename(video_path)
    # delete the extension
    video_id = os.path.splitext(base_name)[0]

    return video_id

# print the number of frames in the video
def print_ldir(ldir):
    simple_ldir = [e[6:10] for e in ldir] 
    print(f'Number of frames in the video: {len(ldir)}, ldir: {simple_ldir}')


# Function to encode images from a directory to base64 strings
def encode_images_to_base64(directory):
    images_base64 = []
    # first print the shape of the video frames
    ldir = os.listdir(directory)
    # IMPORTANT: make sure the frames are sorted in the correct order
    ldir.sort()
    print_ldir(ldir)

    for image_name in ldir:
        image_path = os.path.join(directory, image_name)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            images_base64.append({"image": encoded_string})
    return images_base64


def extract_and_store_frames(video_path, target_dir, max_frames, target_width=None, target_height=None, fps=None):
    """
    Extracts frames from a video and stores them as image files in a target directory.

    Args:
        video_path (str): Path to the input video file.
        target_dir (str): Directory where extracted frames will be saved.
        max_frames (int): Maximum number of frames to extract.
        target_width (int, optional): Desired width for resized frames. If None, original width is preserved.
        target_height (int, optional): Desired height for resized frames. If None, original height is preserved.
        fps (float, optional): Frames per second to sample. If None, frames are sampled uniformly to reach max_frames.
    
    Returns:
        list: List of file paths to the extracted frames.
    """
    # Ensure the target directory exists; if it does, clear it
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Created directory: {target_dir}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video {video_path}, {total_frames} frames, {video_fps} fps, {duration:.2f} seconds")

    max_frames = max(min(max_frames, total_frames // 4), 4)

    # Determine frame indices to extract
    if fps is not None:
        # Sample frames based on specified FPS
        frame_interval = max(1, math.floor(video_fps / fps))
        frame_indices = list(range(0, total_frames, frame_interval))
        frame_indices = frame_indices[:max_frames]
    else:
        # Uniformly sample frames to reach max_frames
        if total_frames < max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int).tolist()

    print(f"Extracting {len(frame_indices)} frames..., max_frames: {max_frames}")

    extracted_frames = []
    for idx, frame_num in enumerate(frame_indices):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_num}")
            continue

        # Resize frame if target dimensions are specified
        if target_width is not None and target_height is not None:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Save frame as image
        frame_filename = f"frame_{idx + 1:04d}.jpg"
        frame_path = os.path.join(target_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)
        # print(f"Saved frame {idx + 1}: {frame_path}")

    cap.release()
    print(f"Finished extracting frames.")

    return extracted_frames


# Function to load video frames using OpenCV -> evenly sample frames
def load_video_by_opencv2(model_dir, video_path, max_frames_num, target_width=None, target_height=None):
    # first extract and store the frames
    video_id = get_video_id(video_path)
    video_frame_folder = f"{clip_video_dir}/{model_dir}/{video_id}"

    results = extract_and_store_frames(video_path, video_frame_folder, max_frames_num, target_width, target_height)

    if results == []:
        images_base64 = []
    else:
        images_base64 = encode_images_to_base64(video_frame_folder)

    return images_base64



# Function to extract frames from video
def load_video(video_path, max_frames_num, fps=1, force_sample=False, target_width=None, target_height=None):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), None, None
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = max(1, round(vr.get_avg_fps() / fps))
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    # Resize frames if target dimensions are specified
    if target_width is not None and target_height is not None:
        resized_frames = []
        for frame in spare_frames:
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            resized_frames.append(resized_frame)
        spare_frames = np.array(resized_frames)

    return spare_frames, frame_time, video_time