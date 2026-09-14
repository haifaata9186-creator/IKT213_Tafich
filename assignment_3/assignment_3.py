import cv2
import numpy as np


#1#define sobel_edge_detection function
def sobel_edge_detection(image):
    #make the image little blurry
    blurred_image = cv2.GaussianBlur(image, (3,3), 0)
    #to find the edge
    sobel = cv2.Sobel(
        blurred_image,
        cv2.CV_64F,
        dx=1,
        dy=1,
        ksize=1
    )

    #convert it to normal image so can save it
    sobel = cv2.convertScaleAbs(sobel)
    return sobel



#2# define canny_edge_detection function
def canny_edge_detection(image, threshold_1, threshold_2):
    blurred_image = cv2.GaussianBlur(image, (3,3), 0)
    canny = cv2.Canny(
        blurred_image,
        threshold_1,
        threshold_2
    )
    return canny


#3#define template function

def template_match(image, template):

    #convert both to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #search for template inside the image,,it gives matric
    result = cv2.matchTemplate(
        image_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    # keep only result with more than 90% similarity.
    threshold = 0.9

    #take the good matches and around red boxes around them
    ##first find all positions that  threshold more than 90%
    locations = np.where(result >= threshold)
    ##the size of template , need it , because the red rectangler has the same size as template
    h, w = template_gray.shape

    ##go through every position matches
    for point in zip(*locations[::-1]):
        cv2.rectangle(
            image,
            point,
            (point[0] + w, point[1] + h),
            (0, 0, 255),
            2
        )

    return image


#4# resize function
def resize( image, scale_factor: int, up_or_down: str):

        steps = int(np.log2(scale_factor))

        for _ in range(steps):
            if up_or_down == "up":
                image = cv2.pyrUp(image)

            elif up_or_down == "down":
                image = cv2.pyrDown(image)

            else:
                raise ValueError("up_or_down must be 'up' or 'down'")

        return image













def main():
    image = cv2.imread("lambo.png")

    # 1#Call the function sobel_edge_detection
    sobel_output = sobel_edge_detection(image)
    # save the
    cv2.imwrite("sobel_output.jpg", sobel_output)

    # show the image
    cv2.imshow("sobel_output", sobel_output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2#Call the function canny_edge_detection
    canny_output = canny_edge_detection (image, 50, 50)
    # save the
    cv2.imwrite("canny_output.jpg", canny_output)

    # show the image
    cv2.imshow("canny_output", canny_output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()






    #3#
    shapes_image = cv2.imread("shapes-1.png")
    template = cv2.imread("shapes_template.jpg")

    template_match_output = template_match(shapes_image, template)
    # save the
    cv2.imwrite("template_match_output.jpg", template_match_output)

    # show the image
    cv2.imshow("template_match_output", template_match_output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #4# resize
    resize_up_result = resize(image, 2, "up")
    cv2.imwrite("resize_up.png", resize_up_result)

    # show the image
    cv2.imshow("resize_up_result", resize_up_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    resize_down_result = resize(image, 2, "down")
    cv2.imwrite("resize_down.png", resize_down_result)

    # show the image
    cv2.imshow("resize_down_result", resize_down_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()