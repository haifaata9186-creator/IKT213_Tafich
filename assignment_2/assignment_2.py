import cv2
import numpy as np

#padding function:
#1#padding add more border arounfd the image, mean more reflecting pixel of the edges, so filter can work on the egdes , and without padding , edges do not have enough pixels neighbours around for image proccessing.
def padding(image, border_width): #define padding function with  two arrguments
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT## to make reflect border pixel not black
    )
    return padded_image


#2#define crop fuction
def crop( image, x_0, x_1,  y_0, y_1):
    croped_image = image[y_0:y_1, x_0:x_1]
    return croped_image

#3#resize function definition
def resize( image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image


#4#manual copy



#create copy function
def copy (image, emptyPictureArray):
    #start copy the pixel
    for y in range(image.shape[0]):# go row by row in the image
        for x in range(image.shape[1]): #go pixel in each row
            emptyPictureArray[y, x] = image[y, x] #copy each pixel
    return emptyPictureArray




#5#
def grayscale(image):


    #convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# 6 # HSV

# create hsv function
def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image





# 7 # Color shifting

def hue_shifted(image, emptyPictureArray, hue):

    # add hue value to every color value
    shifted = image.astype(np.int16) + hue

    # keep all values between 0 and 255
    shifted = np.clip(shifted, 0, 255)

    # put the new pixels into the empty image
    emptyPictureArray[:] = shifted.astype(np.uint8)

    return emptyPictureArray





#8#create smoothing function
def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)#blur the image
    return smoothed_image





#9# rotation function
def rotation(image, rotation_angle):
    #handle angle
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    return rotated_image





def main():
    image = cv2.imread("iris-1.jpg")

    #1# call the function padding and store it in padded_image
    padded_image = padding(image, 100)

    # save the image
    cv2.imwrite("padded_image.png", padded_image)
    # show the image
    cv2.imshow("padded_image", padded_image)
    # keep the image windo open
    cv2.waitKey(0)
    # close window
    cv2.destroyAllWindows()

    #2# to jnow the height and wight of the image
    height, width  = image.shape[:2]
#call the function crop
    croped_image = crop(
      image,
      200,
      width  - 130,
      200,
      height - 130
    )

#save the image
    cv2.imwrite("croped_image.png", croped_image)
#show the image
    cv2.imshow("croped_image", croped_image)
#keep the image windo open
    cv2.waitKey(0)
#close window
    cv2.destroyAllWindows()

#3##call the resize function
    resized_image = resize( image, 200, 200 )
#save the image
    cv2.imwrite("resized_image.png", resized_image)
#show the image
    cv2.imshow("resized_image", resized_image)
#keep the image windo open
    cv2.waitKey(0)
#close window
    cv2.destroyAllWindows()

    #4##get the image size
    height, width, channels = image.shape
#create empty image
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    # call the copy function
    copied_image = copy(image, emptyPictureArray)
    # save the image
    cv2.imwrite("copied_image.png", copied_image)
    # show the image
    cv2.imshow("copied_image", copied_image)
    # keep the image windo open
    cv2.waitKey(0)
    # close window
    cv2.destroyAllWindows()

    #5##call functio
    gray_image = grayscale(image)

#save the image
    cv2.imwrite("gray_image.png", gray_image)
#show the image
    cv2.imshow("gray_image", gray_image)
#keep the image windo open
    cv2.waitKey(0)
#close window
    cv2.destroyAllWindows()


    #6## call the hsv function
    hsv_image = hsv(image)

# save the image
    cv2.imwrite("hsv_image.jpg", hsv_image)

# show the image
    cv2.imshow("HSV Image", hsv_image)

# keep the image window open
    cv2.waitKey(0)

# close the window
    cv2.destroyAllWindows()

#7# # create an empty image
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

# shift all color values by 50
    shifted_image = hue_shifted(image, emptyPictureArray, 50)

# save the image
    cv2.imwrite("hue_shifted_image.jpg", shifted_image)

# show the image
    cv2.imshow("Hue Shifted Image", shifted_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


   #8##call the function smoothed_image
    smoothed_image = smoothing(image)
# save the image
    cv2.imwrite("smoothed_image.jpg", smoothed_image )

# show the image
    cv2.imshow("smoothed_imagel", smoothed_image )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


   #9#Call the function
    rotated_image = rotation(image, 180)
# save the image
    cv2.imwrite("rotated_image.jpg", rotated_image )

# show the image
    cv2.imshow("rotated_image", rotated_image )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()







