import cv2
import numpy as np

class Video:

    def __init__(self, path):
        self.path = path
        # Read videos
        self.fps, self.frames = self.readVideo(path)

    def readVideo(self, path):
        cap = cv2.VideoCapture(path)

        # Get video frames per second
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            raise Exception("Not read file")
        # Initialize frames as an empty list
        frames = []
        for i in range():
            # Read video frame by frame
            ret, frame = cap.read()

            if ret == True:
                # Append frame to the list of frames
                frames.append(frame)
                #cv2.imshow("Frame", frame)
            else:
                break

            key = cv2.waitKey(30)
            # if key q is pressed then break
            if key == 113:
                break
        # Close video file
        cap.release()
        cv2.destroyAllWindows()

        return fps, frames


path = "training/video1.mp4"
if __name__ == '__main__':
    video = Video(path)

    print(len(video.frames))
