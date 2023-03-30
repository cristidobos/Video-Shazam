import cv2
import numpy as np
import pyaudio
import os

import FileReader as file_reader


def get_mp4_files(directory):
    mp4_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            name = os.path.splitext(filename)[0]
            mp4_files.append(name)
    return mp4_files


def print_file_names(mp4_files):
    print("Video library: \n")
    index = 0
    for file_name in mp4_files:
        print("index " + str(index) + " -> " + file_name)
        index += 1
    print("\n----------------------\n")


path_dataset = 'resource/lib/publicdata/Code/dataset/'
path_training = 'training/recognition/'
dataset_files = get_mp4_files(path_dataset)
# print_file_names(dataset_files)

"""
Takes as an input the path to the cropped video, and returns the name of the best match

input: path of the cropped video
output: string with the name of the best match
 
"""


def recognize_video(video):
    print(len(video.frames))


def iterate_over_dataset(dataset_names, given_video, given_audio):
    index = 0
    for file_name in dataset_names:
        video = file_reader.VideoReader(path_dataset + file_name + ".mp4")
        # -------- play video
        # video.display_video()

        audio = file_reader.AudioReader(path_dataset + file_name + ".wav")
        # -------- play audio
        # audio.play_audio()
        print("FINISHED reading DATASET sample")

        index += 1
        if index >= 1:
            break


train_video = file_reader.VideoReader(path_training + "video1" + ".mp4")
train_audio = file_reader.AudioReader(path_training + "video1" + ".wav")
print("FINISHED reading TRAINING sample")
iterate_over_dataset(dataset_files, train_video, train_audio)

