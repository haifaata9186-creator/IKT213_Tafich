import cv2
import numpy as np


#1# define Harris Corner Detection
def harris_corner_detection(reference_image):

    ##need for Harris to make image gray
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    ##need for Harris to convert image to float
    gray = gray.astype("float32")

    ##to find the corner scores
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    ##we want a bigger dot so we can see it easily
    corners = cv2.dilate(corners, None)

    ##make a copy of image
    result = reference_image.copy()

    ##colour a pixel red if its corner score is high enough
    result[corners > 0.01 * corners.max()] = [0, 0, 255]

    ##save the image with the red corner marks
    cv2.imwrite("harris_corners.jpg", result)

    ##return the image with corner marks
    return result


#2# adjust align_this.jpg so it lines up with reference_img.png
def align_images(image_to_align, reference_image,
                 max_features=1500, good_match_percent=0.15):

    ##convert both images to gray
    gray_to_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    ##create the ORB tool to find feature points
    orb = cv2.ORB_create(max_features)

    ##ORB finds points and their descriptions in the first image
    keypoints1, descriptors1 = orb.detectAndCompute(gray_to_align, None)

    ##ORB finds points and their descriptions in the reference image
    keypoints2, descriptors2 = orb.detectAndCompute(gray_reference, None)

    ##check that ORB found features in both images
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Could not find features in one or both images.")

    ##prepare a tool to compare the feature descriptions
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ##compare the descriptions and find matching pairs
    matches = matcher.match(descriptors1, descriptors2)

    ##sort the matches from most similar to least similar
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    ##calculate how many matches to keep: the best 15%
    num_good_matches = int(len(matches) * good_match_percent)

    ##keep only the best 15% of matches
    matches = matches[:num_good_matches]

    ##we need at least four matching point pairs
    if len(matches) < 4:
        raise ValueError("Not enough matches to align the images.")

    ##show both images side by side with lines between matched points
    matches_image = cv2.drawMatches(
        image_to_align, keypoints1,
        reference_image, keypoints2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    ##save the image with matching lines
    cv2.imwrite("matches.png", matches_image)

    ##create an empty table for points in the first image
    points1 = np.zeros((len(matches), 2), dtype=np.float32)

    ##create an empty table for points in the reference image
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    ##fill both tables with x and y coordinates of each matched pair
    for i, match in enumerate(matches):
        points1[i] = keypoints1[match.queryIdx].pt
        points2[i] = keypoints2[match.trainIdx].pt

    ##calculate the transformation and use RANSAC to reject wrong matches
    homography, mask = cv2.findHomography(
        points1, points2, cv2.RANSAC
    )

    ##check that the transformation was found
    if homography is None:
        raise ValueError("Could not calculate the image transformation.")

    ##get the height and width of the reference image
    height, width = reference_image.shape[:2]

    ##apply the transformation to align the first image
    aligned_image = cv2.warpPerspective(
        image_to_align, homography, (width, height)
    )

    ##save the aligned image
    cv2.imwrite("aligned.png", aligned_image)

    ##return both result images
    return aligned_image, matches_image


def main():

    ##load both images
    image_to_align = cv2.imread("align_this.jpg")
    reference_image = cv2.imread("reference_img.png")

    ##check that both images were loaded
    if image_to_align is None or reference_image is None:
        print("Could not open one or both images. Check the filenames and folder.")
        return

    #1# call Harris Corner Detection
    harris_image = harris_corner_detection(reference_image)

    ##show the image with red corner marks
    cv2.imshow("Harris corners", harris_image)

    ##wait for a key and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #2# call image alignment using ORB
    aligned_image, matches_image = align_images(
        image_to_align, reference_image
    )

    ##show the reference image to compare
    cv2.imshow("Reference image", reference_image)

    ##show the aligned image
    cv2.imshow("Aligned image", aligned_image)

    ##show the matching points
    cv2.imshow("Matches", matches_image)

    ##wait for a key and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


##run main when this file is run directly
if __name__ == "__main__":
    main()
