#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import glob
 

def get_intrinsic():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
     
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * 0.25
     
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
     
    # images = glob.glob('/home/nvidia/f1tenth_ws/src/vision/lab-7-vision-lab-team-7/calibration/*.png')
    images = glob.glob('calibration/*.png')
     
    for idx, fname in enumerate(images):

        print(idx)

        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)    
     
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx
    
def get_bottom_right_orange(filename, lower_orange):

    # Read the image
    image = cv.imread(filename)

    # Convert BGR to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define range of orange color in HSV
    lower_orange = np.array(lower_orange)
    upper_orange = np.array([20, 255, 255])

    # Threshold the HSV image to get only orange colors
    mask = cv.inRange(hsv, lower_orange, upper_orange)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find the most bottom-right point
    max_x = 0
    max_y = 0
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if x > max_x or (x == max_x and y > max_y):
                max_x = x
                max_y = y

    # Print the coordinates of the most bottom-right orange pixe
    marked_image = cv.circle(image, (max_x, max_y), 5, (255, 0, 0), -1)
    cv.imshow('image', marked_image)   
    cv.waitKey(0)
    cv.destroyAllWindows()
    return max_x, max_y


def get_mounting_height(x_car_distance=0.4, camera_matrix=np.eye(3, 3), y_pixel=0):
    f_y = camera_matrix[1,1]
    y_0 = camera_matrix[1,2]
    camera_height = (y_pixel- y_0)*x_car_distance/f_y
    return camera_height 

def pixel_to_dist(pixel_coords, camera_matrix, mounting_height):
    camera_coords = np.array([pixel_coords[0], pixel_coords[1], 1])

def get_uknown_coord(camera_matrix, H_mount):
    x_pixel, y_pixel = get_bottom_right_orange('resource/cone_unknown.png', lower_orange=[0, 184, 184])
    print(x_pixel,y_pixel)
    f_x = camera_matrix[0,0]
    f_y = camera_matrix[1,1]
    x_0 = camera_matrix[0,2]
    y_0 = camera_matrix[1,2]
    x_car = f_y*(H_mount)/(y_pixel - y_0)
    y_car = (x_pixel - x_0)/f_x*x_car
    return x_car, y_car


def get_green_coords(image, min_green=(30, 50, 50), max_green=(90, 255, 255)):
    '''
    Gets list of green points in image

    '''

    # Convert image to HSV color space
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array(min_green)
    upper_green = np.array(max_green)

    # Threshold the HSV image to get only green colors
    green_mask = cv.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv.findContours(green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw green points on the original image
    green_points = []
    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            green_points.append([cx, cy])
    
    green_points = np.array(green_points)

    return green_points

def get_unknown_coords_batch(points, camera_matrix, H_mount):
    '''
    Converts list of camera pixels into list of car coordinates
    based on camera mounting height (assuming camera facing in direction of car)
    '''

    # Gather params from camera matrix
    f_x = camera_matrix[0,0]
    f_y = camera_matrix[1,1]
    x_0 = camera_matrix[0,2]
    y_0 = camera_matrix[1,2]

    # Calculate x and y coods
    x_car = f_y * (H_mount) / (points[:, 1] - y_0)
    y_car = (points[:, 0] - x_0)/f_x * x_car
    
    return x_car, y_car
    

def main():
    camera_matrix = get_intrinsic()
    x_pixel, y_pixel = get_bottom_right_orange('resource/cone_x40cm.png', lower_orange=[0, 200, 200])
    H_mount = get_mounting_height(0.40, camera_matrix=camera_matrix, y_pixel=y_pixel)
    # print(camera_matrix)
    # print(H_mount)
    x_car, y_car = get_uknown_coord(camera_matrix, H_mount)
    print(x_car, y_car)
    cv.destroyAllWindows()
 
if __name__ == '__main__':
    main()
