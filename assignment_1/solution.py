import cv2


def print_image_information(image):
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)
image= cv2.imread("iris-1.jpg")
print_image_information(image)

#open the default camera
cam=cv2.VideoCapture(0)

#get camera information
fps = cam.get(cv2.CAP_PROP_FPS)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

# Save the information to a text file
with open("solutions/camera_outputs.txt", "w") as file:
    file.write(f"fps: {fps}\n")
    file.write(f"height: {height}\n")
    file.write(f"width: {width}\n")

# Close the camera
cam.release()


