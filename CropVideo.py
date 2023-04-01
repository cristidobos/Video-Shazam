import cv2
import numpy as np
import matplotlib.pyplot as plt
import VideoReader


def crop_video(video):
    corners = test_shit(video)
    # Get the video properties
    fps = video.fps
    height, width, channels = video.frames[0].shape

    # Create a list to store the cropped frames
    cropped_frames = []

    # Loop over each frame of the video
    for frame in video.frames:
        # Crop the frame using the specified corners
        cropped_frame = crop_image(frame, corners)

        # Append the cropped frame to the list
        cropped_frames.append(cropped_frame)

    # Create a new video object with the cropped frames
    cropped_video = type(video)(fps=fps, frames=np.array(cropped_frames))

    return cropped_video


def play_video(video):
    # Create a window to display the video
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

    # Loop over each frame of the video
    for frame in video.frames:
        # Display the frame
        cv2.imshow("Video Player", frame)

        # Wait for a key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Destroy the window
    cv2.destroyAllWindows()


# def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
#     # Convert the image to grayscale
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = image
#     # Apply a Gaussian blur to the image to reduce noise
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     # Compute the Harris corner response function
#     harris = cv2.cornerHarris(gray, block_size, ksize, k)
#
#     # Normalize the response function
#     cv2.normalize(harris, harris, 0, 255, cv2.NORM_MINMAX)
#
#     plt.imshow(harris, cmap='gray')
#     plt.show()
#
#     # Threshold the response function to extract corner points
#     corners = np.argwhere(harris > threshold * harris.max())
#     corners = np.float32([corners])
#
#     # Perform non-maximum suppression on the corner points
#     corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
#                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
#
#     # Draw circles around the corner points
#     for x, y in corners[0]:
#         # x, y = corner
#         cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), 2)
#
#     return image
#

def adjustContrast(frame):
    # Apply gaussian blur to remove noise
    blur = cv2.GaussianBlur(frame, (5, 5), 1)

    # Convert image to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Apply histogram equalization for contrast enhancement
    # image_equalized = cv2.equalizeHist(gray)

    return gray


# ----------------testing-------------------------------------------
def compute_frame_gradient(prev_frame, curr_frame):
    # Convert frames to grayscale
    # prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute the gradient between the pixels of the two frames
    gradient = cv2.absdiff(curr_frame, prev_frame)

    return gradient


def test_shit(video):
    # old_gradient = compute_frame_gradient(adjustContrast(video.frame[0]), adjustContrast(video.frame[1]))
    final_gradient = np.zeros((len(video.frames[0]), len(video.frames[0][0])))

    count = 0
    for i in range(1, len(video.frames)):
        frame1 = video.frames[i - 1]
        frame2 = video.frames[i]

        frame1 = adjustContrast(frame1)
        frame2 = adjustContrast(frame2)

        gradient = compute_frame_gradient(frame1, frame2)
        final_gradient += gradient
        count += 1

    final_gradient *= 1 / count
    plt.imshow(gradient, cmap='gray')
    plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    grad_closing = cv2.morphologyEx(final_gradient, cv2.MORPH_CLOSE, kernel, iterations=1)
    plt.imshow(grad_closing, cmap='gray')
    plt.show()

    normalize_grad = grad_closing / np.max(grad_closing) * 255
    plt.imshow(normalize_grad, cmap='gray')
    plt.show()
    normalize_grad = remove_thin_lines(normalize_grad)

    # filtered_grad = np.where(normalize_grad < 100, 0, grad_closing)
    img_uint8 = cv2.convertScaleAbs(normalize_grad)
    plt.imshow(img_uint8, cmap='gray')
    plt.show()

    _, filtered_grad = cv2.threshold(img_uint8, 100, 255, cv2.THRESH_BINARY)
    plt.imshow(filtered_grad, cmap='gray')
    plt.show()

    # modified_grad = remove_thin_lines(filtered_grad)
    # plt.imshow(modified_grad, cmap='gray')
    # plt.show()
    # filtered_grad = modified_grad

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(filtered_grad, kernel, iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=1)
    filtered_grad = dilated
    plt.imshow(filtered_grad, cmap='gray')
    plt.show()

    corners = find_rectangle_corners(filtered_grad)
    # print(corners)
    # cropped_image = crop_image(filtered_grad, corners)
    # plt.imshow(cropped_image, cmap='gray')
    # plt.show()

    return corners


def find_rectangle_corners(image):
    # Find the coordinates of all non-zero pixels in the binary image
    coords = np.column_stack(np.where(image > 0))

    # Find the top-left and bottom-right corners of the rectangle
    tl_corner = coords.min(axis=0)
    br_corner = coords.max(axis=0)

    # Find the top-right and bottom-left corners of the rectangle
    tr_corner = np.array([br_corner[0], tl_corner[1]])
    bl_corner = np.array([tl_corner[0], br_corner[1]])

    # Return the coordinates of all four corners of the rectangle
    return [tl_corner.tolist(), tr_corner.tolist(), br_corner.tolist(), bl_corner.tolist()]


def crop_image(image, corners, padding=0.01):
    # Convert the corner coordinates to integers
    corners = np.array(corners, dtype=np.int32)

    # Find the minimum and maximum x and y coordinates of the corners
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    # Compute the width and height of the bounding box
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Add padding to the width and height
    pad_x = int(padding * width)
    pad_y = int(padding * height)

    # Adjust the coordinates of the bounding box by the padding
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image.shape[0] - 1, x_max + pad_x)
    y_max = min(image.shape[1] - 1, y_max + pad_y)

    # Crop the image to the adjusted bounding box
    cropped_image = image[x_min:x_max + 1, y_min:y_max + 1]

    return cropped_image


def remove_thin_lines(image):
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=1)
    return dilated

# --------------- not used--------------------
# def harris_corner_detection(image, image_rgb, block_size=2, ksize=3, k=0.04, threshold=0.01):
#     # Convert the image to grayscale
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply a Gaussian blur to the image to reduce noise
#     gray = image.copy()
#
#     # Compute the Harris corner response function
#     harris = cv2.cornerHarris(gray, block_size, ksize, k)
#
#     # Normalize the response function
#     cv2.normalize(harris, harris, 0, 255, cv2.NORM_MINMAX)
#
#     # Draw the detected corners onto the image
#     corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
#     corners = np.int0(corners)
#     for corner in corners:
#         x, y = corner.ravel()
#         cv2.circle(image_rgb, (x, y), 3, (0, 0, 255), -1)
#     plt.imshow(image_rgb)
#     plt.show()
#
#     return image
#
# def cropFrame(contours, original):
#     # Select the largest contour
#     plot_contours(original, contours)
#     contours = filter_contours(contours)
#
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     display = contours[0]
#
#     # Find bounding box of object
#     x, y, w, h = cv2.boundingRect(display)
#     print("RATIO: " + str(w / h))
#     # cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 3)
#     # Crop the frame around bounding box
#     cropped = original[y: y + h, x: x + w, :]
#     plt.imshow(cropped)
#     plt.show()
#     # return original
#     return cropped
# def filter_contours(contours):
#     filtered_contours = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         ratio = w / h
#         if w >= 100 and h >= 100:
#             filtered_contours.append(contour)
#     return filtered_contours
# def plot_contours(image, contours):
#     """
#     Plots the original grayscale image with all the contours from the input 'contours' vector
#     drawn on it using OpenCV's cv2.drawContours function.
#     """
#     # Draw all the contours on a copy of the original image
#     # image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     image_with_contours = image.copy()
#     cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
#
#     # Plot the original image and the image with contours using Matplotlib
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     ax[0].imshow(image)
#     ax[0].set_title('Original Image')
#     ax[1].imshow(image_with_contours)
#     ax[1].set_title('Image with Contours')
#
#     # Show the plot
#     plt.show()
# def findCountours(image):
#     # Perform adaptive thresholding to get a binary image
#     # thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
#     # inverted_img = cv2.bitwise_not(image)
#
#     ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
#
#     # thresh = cv2.Canny(image, 100, 200)
#     # Find contours present in image
#     cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#     return cnts

# def crop_video(video):
#     # The list of frames
#     frames = video.frames
#
#     # Iterate through each frame
#     for i in range(len(frames)):
#         frame = frames[i]
#         # # Adjust contrast of frame
#         adjusted = adjustContrast(frame)
#         plt.imshow(adjusted, cmap='gray')
#         plt.show()
#         # # Find countours of objects within frame
#         # contours = findCountours(adjusted)
#         #
#         # # Crop the frame given the largest contour
#         # cropped = cropFrame(contours, frame)
#         #
#         # # Replace with cropped
#         cropped = harris_corner_detection(adjusted, frame)
#         plt.imshow(cropped, cmap='gray')
#         plt.show()
#         frames[i] = cropped
#         if i >= 0:
#             break
#     return video
