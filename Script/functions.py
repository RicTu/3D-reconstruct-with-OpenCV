import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint
import tkinter as tk
from tkinter import filedialog
import glob
import numpy as np
import pandas as pd
import math

def read_video_from_certain_frame(video_path, start_frame):
    """
    Get VideoCapture obect start from certain frame.

    Inputs:
    - video_path (string): video path
    - start_frame (int): start frame

    Outputs:
    - cap (VideoCapture): VideoCapture obect start from certain frame
    """

    cap = cv2.VideoCapture(video_path)
    print("frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    print("fps:", int(cap.get(cv2.CAP_PROP_FPS)))

    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    print(f"video length: {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.2f} seconds")

    cap.set(1, start_frame)

    return cap


def screenshot_chessboard(left_video_path, right_video_path, left_start_frame, right_start_frame, num_img, period):
    """
    Screenshot the videos to get images each of them has a gap with some number of frames.

    Inputs:
    - left_video_path (string): video path for left camera
    - right_video_path (string): video path for right camera
    - left_start_frame (int): start frame for left camera
    - right_start_frame (int): start frame for right camera
    - num_img (int): number of screenshots want to get
    - period (int): frame number between two screenshots

    Outputs:
    results (list): a list has two list inside it, containing left and right images in the same order
    """

    # Read left and right video from given frame
    left_cap = read_video_from_certain_frame(left_video_path, left_start_frame)
    right_cap = read_video_from_certain_frame(
        right_video_path, right_start_frame)

    # Iterate the with a given step to get the screenshots
    img_cnt = 0
    cur_frame_left, cur_frame_right = left_start_frame, right_start_frame
    results = [[], []]
    while cur_frame_left < int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT)) and cur_frame_right < int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT)) and img_cnt < num_img:
        left_cap.set(1, cur_frame_left)
        success, left_frame = left_cap.read()

        if not success:
            print("left frame read failed")
            return
        # left_frame = cv2.flip(left_frame, -1)

        right_cap.set(1, cur_frame_right)
        success, right_frame = right_cap.read()

        if not success:
            print("right frame read failed")
            return

        # right_frame = cv2.flip(right_frame, -1)

        # cv2.imshow("imgleft", left_frame)
        # cv2.imshow("imgright", right_frame)
        # cv2.waitKey(0)


        results[0].append(left_frame)
        results[1].append(right_frame)
        cur_frame_left, cur_frame_right = cur_frame_left + period, cur_frame_right + period
        img_cnt += 1

    return results

def masking_for_one_ccp(pos, image_path, start_frame, offset, mask_num):
    """
    Mask the chessboards to make each image with exactly one chessboard exists.

    Inputs:
    - pos (string): position of the camera, can be "left" or "right"
    - video_path (string): video path
    - start_frame (int): start frame
    - offset (int): offset frame numbers from start_frame
    - mask_num (int): number of chessboards in a frame

    Outputs:
    - results (list): list of masked images
    """

    # Read image from given frame number and offset
    image = cv2.imread(image_path)
    
    """
    # Flip the image for right view to get the correct order
    if pos == "right":
        frame = cv2.flip(frame, -1)
    """
    # Select the chessboards
    bboxes = []
    while len(bboxes) < mask_num:
        bbox = cv2.selectROI("Select chessboards", image)
        p1, p2 = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        bboxes.append([p1, p2])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break

    cv2.destroyAllWindows()

    results = []

    # Mask the chessboards based on the selection above
    for i in range(mask_num):
        frame_temp = image.copy()

        for j in range(mask_num):
            if i != j:
                frame_temp[bboxes[j][0][1]:bboxes[j][1][1],
                           bboxes[j][0][0]:bboxes[j][1][0]] = 255

        results.append(frame_temp)

    return results

def calibrate(images, cbrow, cbcol, pos, num):
    """
    Calibrate the camera to get intrinsic matrix.

    Inputs:
    - images (list): a list of image objects
    - cbrow (int): number of rows of the chessboard
    - cbcol (int): number of columns of the chessboard
    - pos (string): position of the camera, can be "left" or "right"
    - num (int): number of images used for calibration

    Returns:
    - ret (float): re-projection error
    - mtx (numpy.array): intrinsic matrix
    - dist (numpy.array): distortion coefficients
    - rvecs (numpy.array): rotation vector
    - tvecs (numpy.array): translation vector
    - objpoints (dict): 3D object points
    - imgpoints (dict): 2D image points
    """

    # termination criteria
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow*cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = {}  # 3d point in real world space
    imgpoints = {}  # 2d points in image plane.
    objpoints_list = []
    imgpoints_list = []
    success_images = []
    for img_num, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            img_num = str(img_num)
            success_images.append(img_num)

            objpoints[img_num] = objp
            objpoints_list.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints[img_num] = corners2
            imgpoints_list.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
            cv2.namedWindow(img_num, cv2.WINDOW_NORMAL)
            cv2.imshow(img_num, img)
            # cv2.imwrite(f"D:/test/{img_num}.jpg", img)
            cv2.waitKey(200)
        else:
            print(f"{img_num} failed to find corners.")

    cv2.destroyAllWindows()

    if images is not []:
        # Use the results above to calibrate the camera to get intrinsic matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_list, imgpoints_list, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, success_images

def ccp_for_one_calibration(left_video,right_video,left_ccp,right_ccp):
    # Note: the "left" and "right" is the relative position, not absolute position

    ### Some configurations ###
    # chessboard for grid
    cbrow, cbcol = 8, 6
    # video resolution
    image_size = (1280, 720)
    # left_video, right_video = "../Camera_calibrate/calibrate_R_sync.mp4", \
    #                           "../Camera_calibrate/calibrate_B_sync.mp4"

    #left_ruler, right_ruler = "./videos/up_right_1.mp4", "./videos/up_right_2.mp4"
    left_ruler, right_ruler=left_ccp,right_ccp
    # offset of two videos (the frame we are interested after synchronizing)
    #video_offset, chess_offset, ruler_offset = 2500, 0, 0

    video_offset, chess_offset, ruler_offset = 0, 0, 0
    # number of chessboard appear on a frame
    mask_num = 4
    # error tolerance of calibration
    threshold = 1.0

    ### Estimate intrinsic matrices and distortion coefficients for two cameras ###
    # Synchronize two videos
    print("synchronizing")
    # left_frame_sync, right_frame_sync = sync_videos(left_video, right_video, 500, 1100)
    left_frame_sync=0
    right_frame_sync=0
    # Screenshot the videos to get some chessboard images for calibration
    print("screenshot_chessboard...")
    calib_images = screenshot_chessboard(
        left_video, right_video, left_frame_sync+video_offset, right_frame_sync+video_offset, 20, 100)

    # Calibrate each cameras
    print("calibrating..")
    ret1, mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1, success_images1 = calibrate(
        calib_images[0], cbrow, cbcol, "left", len(calib_images[0]))
    ret2, mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2, success_images2 = calibrate(
        calib_images[1], cbrow, cbcol, "right", len(calib_images[1]))

    # Print the result of calibration
    #if verbose:
    print("-"*77)
    print("re-projection error (pixel)")
    print(ret1)
    print(ret2)
    print("-"*77)
    print(f"Intrinsic matrix (shape={mtx1.shape})")
    print(mtx1)
    print(mtx2)
    print("-"*77)
    print(f"Distortion coefficients (shape={dist1.shape})")
    print(dist1)
    print(dist2)
    print("-"*77)

    # Check the result of calibration
    if ret1 < threshold and ret2 < threshold:
        print("Both camera achieve great results for calibration.")
    else:
        print("At least one of the cameras has unacceptable calibration result.")

    ### Get new projection matrices for both cameras ###
    """# Synchronize two videos
    left_frame_sync, right_frame_sync = sync_videos(
        left_ruler, right_ruler, 1, 1)
    """
    # Mask the chessboard for both cameras
    left_mask_images = masking_for_one_ccp(
        "left", left_ruler, left_frame_sync, chess_offset, mask_num)
    right_mask_images = masking_for_one_ccp(
        "right", right_ruler, right_frame_sync, chess_offset, mask_num)

    # Using the masked chessboard images to calibrate two cameras
    _, _, _, _, _, objpoints1, imgpoints1, success_images1 = calibrate(
        left_mask_images, cbrow, cbcol, "left", mask_num)
    _, _, _, _, _, objpoints2, imgpoints2, success_images2 = calibrate(
        right_mask_images, cbrow, cbcol, "right", mask_num)

    # Use the images which both camera successfully found the corners
    intersect_nums = [num for num in success_images1 if num in success_images2]
    print(intersect_nums)
    objpoints1, objpoints2 = [objpoints1[n] for n in intersect_nums], [
        objpoints2[n] for n in intersect_nums]
    imgpoints1, imgpoints2 = [imgpoints1[n] for n in intersect_nums], [
        imgpoints2[n] for n in intersect_nums]

    # Stereo calibrate two cameras to get new projection matrices for two cameras and the translation matrix between two cameras
    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 10, 1e-5)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints1, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, image_size, criteria=stereocalib_criteria, flags=cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_ZERO_TANGENT_DIST)
    # Print the result of stereo calibration
    #if verbose:
    print("Stereo Calibrate")
    print("-"*77)
    print("Error")
    print(retval)

    # Stereo rectification to make two image plane parallel, this will get a rectified projection matrix for both two cameras
    global R1
    global R2
    global P1
    global P2
    R1, R2, P1, P2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2,
                                       image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)[:4]
    #if verbose:
    print("Stereo Rectify")
    print("-"*77)
    print(f"Rectification transform (rotation matrix) for the first camera (shape={R1.shape})")
    print(R1)
    print("-"*77)
    print(f"Rectification transform (rotation matrix) for the second camera (shape={R2.shape})")
    print(R2)
    print("-"*77)
    print(f"Projection matrix in the new (rectified) coordinate systems for the first camera (shape={P1.shape})")
    print(P1)
    print("-"*77)
    print(f"Projection matrix in the new (rectified) coordinate systems for the second camera (shape={P2.shape})")
    print(P2)  
    return mtx1,dist1,mtx2,dist2,R1,R2,P1,P2,objpoints1,imgpoints1,imgpoints2


def start_camera_calibrate(pair):
    #loaing the camera calibration videos and calculation the stereo transformation matrix-------S7-camera-------------
    prefix = pair.split("_")

    left_video_path = f"../Camera_calibrate/calibrate_{prefix[0]}_sync.mp4"
    right_video_path = f"../Camera_calibrate/calibrate_{prefix[1]}_sync.mp4"

    left_ccp_path = f"../Camera_calibrate/CCP_img_{prefix[0]}.png"
    right_ccp_path = f"../Camera_calibrate/CCP_img_{prefix[1]}.png"
    # Note: the "left" and "right" is the relative position, not absolute position
    # e.g. the left means image 1, the right means image 2 

    print(left_video_path)
    mtx1,dist1,mtx2,dist2,R1,R2,P1,P2,objpoints1,imgpoints1,imgpoints2=ccp_for_one_calibration(
        left_video_path,right_video_path,left_ccp_path,right_ccp_path)

    np.save(f"../Camera_matrix/{pair}_mtx1.npy", mtx1)
    np.save(f"../Camera_matrix/{pair}_dist1.npy", dist1)
    np.save(f"../Camera_matrix/{pair}_mtx.npy", mtx2)
    np.save(f"../Camera_matrix/{pair}_dist2.npy", dist2)
    np.save(f"../Camera_matrix/{pair}_R1.npy", R1)
    np.save(f"../Camera_matrix/{pair}_R2.npy", R2)
    np.save(f"../Camera_matrix/{pair}_P1.npy", P1)
    np.save(f"../Camera_matrix/{pair}_P2.npy", P2)


    ###


def  calculate_3D_point(mtx1,dist1,mtx2,dist2,R1,R2,P1,P2,leftpoint,rightpoint):
    projPoints1 = np.array(leftpoint).reshape(-1, 1, 2)
    projPoints2 = np.array(rightpoint).reshape(-1, 1, 2)

    # Undistort the 2D points
    projPoints1 = cv2.undistortPoints(
        src=projPoints1, cameraMatrix=mtx1, distCoeffs=dist1, R=R1, P=P1)
    projPoints2 = cv2.undistortPoints(
        src=projPoints2, cameraMatrix=mtx2, distCoeffs=dist2, R=R2, P=P2)

    # Triangulate the 2D points to get corresponding 3D points (actually 4D because using homogeneous coordinates)
    points4d = []
    projPoints1, projPoints2 = projPoints1.reshape(
        -1, 2).T, projPoints2.reshape(-1, 2).T
    points4d.append(cv2.triangulatePoints(P1, P2, projPoints1, projPoints2))

    points4d = np.array(points4d)
    # np.save("./points4d.npy", points4d)

    # Transform points from homogeneous coordinates to cartesian coordinate system (4D to 3D)
    points3d = []
    for i in range(points4d.shape[0]):
        points3d.append(np.array([points4d[i, :, j]
                                for j in range(points4d.shape[-1])]))
    pi = []
    for each_points3d in points3d:
        pi.append(np.array([each_points3d[i][:-1] / each_points3d[i][-1]
                            for i in range(each_points3d.shape[0])]))

    # Save results
    dist = []
    for i in range(len(pi[i]) // 2):
        dist.append(np.linalg.norm(pi[0][i*2]-pi[0][i*2+1]) * 10)

    return pi

def load_cam_info(cam_pair):  # input "B_L" or "B_R" or "R_L"
    mtx1 = np.load(f"../Camera_matrix/{cam_pair}_mtx1.npy")                               
    dist1 = np.load(f"../Camera_matrix/{cam_pair}_dist1.npy")                             
    mtx2 = np.load(f"../Camera_matrix/{cam_pair}_mtx.npy")                                
    dist2 = np.load(f"../Camera_matrix/{cam_pair}_dist2.npy")                             
    R1 = np.load(f"../Camera_matrix/{cam_pair}_R1.npy")                                   
    R2 = np.load(f"../Camera_matrix/{cam_pair}_R2.npy")                                   
    P1 = np.load(f"../Camera_matrix/{cam_pair}_P1.npy")                                   
    P2 = np.load(f"../Camera_matrix/{cam_pair}_P2.npy") 
    return mtx1,dist1,mtx2,dist2,R1,R2,P1,P2

def retify(point):
    cx = 720/2
    cy = 1280/2
    temp = point
    old_coor_offset = [temp[0]-cx, temp[1]-cy]
    return [cy - old_coor_offset[1], cx + old_coor_offset[0]]


def detect_chessboard(img_path):
#     img=cv2.imread(img_path)
    img=cv2.imread(img_path)
    if img.shape == (1280, 720, 3):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.rectangle(img, (0, 0), (1280, 360), (0,0,0), -1)
    cv2.rectangle(img, (800, 0), (1280, 720), (0,0,0), -1)

    cbrow, cbcol = 8, 6 # chessboard for grid
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imshow("img",img)
        cv2.waitKey(200)
        cv2.destroyAllWindows()
        # plt.imshow(img)
        # plt.show()
        return corners2
    else:
        print(f"Image fail to detect")
    print("Go through all imgs, can't detect corner")
    
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def get_transformation_matrix(point3d): ## input 3d points of chessboard
    chessboardpoints=[
            [point3d[40][0], point3d[40][1], point3d[40][2]],
            [point3d[41][0], point3d[41][1], point3d[41][2]],
            [point3d[32][0], point3d[32][1], point3d[32][2]]
            ]
    cp0, cp1, cp2 = chessboardpoints
    cx0, cy0, cz0 = cp0
    cx1, cy1, cz1 = cp1
    cx2, cy2, cz2 = cp2
    # print(chessboardpoints)

    cux, cuy, cuz = cu = [cx1-cx0, cy1-cy0, cz1-cz0] #Vector c0->c1
    normalized_cu = cu/np.linalg.norm(cu)
    cvx, cvy, cvz = cv = [cx2-cx0, cy2-cy0, cz2-cz0] #Vector c0->c2
    # normalized_cv = cv/np.linalg.norm(cv)
    c_u_cross_v = np.cross(cu,cv) #Normal vector 
    # c_u_cross_v = [cuy*cvz - cuz*cvy, cuz*cvx - cux*cvz, cux*cvy - cuy*cvx] #Normal vector
    c_a, c_b, c_c = c_u_cross_v
    normalized_cuv = c_u_cross_v/np.linalg.norm(c_u_cross_v)
    n_cvx, n_cvy, n_cvz = new_cv=np.cross(cu,c_u_cross_v)
    normalized_cv = new_cv/np.linalg.norm(new_cv)

    # original coordinate system
    original_system=[[1,0,0],
                [0,1,0],
                [0,0,1]]
    m11=np.dot(normalized_cv, original_system[0])
    m12=np.dot(normalized_cv, original_system[1])
    m13=np.dot(normalized_cv, original_system[2])
    #
    m21=np.dot(normalized_cu, original_system[0])
    m22=np.dot(normalized_cu, original_system[1])
    m23=np.dot(normalized_cu, original_system[2])
    #
    m31=np.dot(normalized_cuv, original_system[0])
    m32=np.dot(normalized_cuv, original_system[1])
    m33=np.dot(normalized_cuv, original_system[2])
    transformation_matrix=np.array(( [m11, m12, m13],
                                    [m21, m22, m23],
                                    [m31, m32, m33],
    ))
    return transformation_matrix


def calculate_3d_datas_and_feature(information, pair):
    in1_L = np.load(f"../Result/{information[0]}.npy")
    th1_L = np.load(f"../Result/{information[1]}.npy")

    in1_B = np.load(f"../Result/{information[2]}.npy")
    th1_B = np.load(f"../Result/{information[3]}.npy")


    mtx1, dist1, mtx2, dist2, R1, R2, P1, P2 = load_cam_info(pair) 

    prefix = pair.split("_")
    left_img_path = f"../Camera_calibrate/CCP_img_{prefix[0]}.png"
    back_img_path = f"../Camera_calibrate/CCP_img_{prefix[1]}.png"


    # left_check_img = check_coor(left_img_path, th1_L[0], back = False)
    # back_check_img = check_coor(back_img_path, coorB, th1_B[0], back = True)

    # t, arr = plt.subplots(1,2,figsize=(12,5))
    # arr[0].imshow(left_check_img)
    # arr[1].imshow(back_check_img)
    # plt.savefig("../Result/checkCoor.png")
    # if show_fig:
    #     plt.show()

    point3d_th, point3d_in, dt = [], [], []
    for i in range(len(th1_L)):   ## using left and back to calculate 3d object points 
        point3d_in.append(calculate_3D_point(mtx1, dist1, mtx2, dist2, R1, R2, P1, P2, 
                                        in1_L[i], retify(in1_B[i]))[0][0]) 
        # point3d_th.append(calculate_3D_point(mtx1, dist1, mtx2, dist2, R1, R2, P1, P2, 
        #                                     th1_L[i]+coorL, retify(th1_B[i], coorB) )[0][0])
        dt.append(i/240)
        
    point3d_in = np.array(point3d_in); point3d_th = np.array(point3d_th)
    print(f"index length: {len(point3d_in)}, thumb length: {len(point3d_th)}")    

    df = pd.DataFrame()
    df["time"] = dt
    df["index_x"] = point3d_in[:,0]; df["index_y"] = point3d_in[:,1]; df["index_z"] = point3d_in[:,2]
    # df["thumb_x"] = point3d_th[:,0]; df["thumb_y"] = point3d_th[:,1]; df["thumb_z"] = point3d_th[:,2]
    # df["dis"] = np.sum(np.sqrt((point3d_in - point3d_th)**2), axis = 1)
    # df.to_csv("../Result/3D_result_pixel.csv", index=False)

    # pixel convert to cm and coordinate transformation
    chess_points_l = detect_chessboard(left_img_path)
    chess_points_b = detect_chessboard(back_img_path)
    chess_corner = []
    for i in range(len(chess_points_b)):
        chess_corner.append(calculate_3D_point(mtx1, dist1, mtx2, dist2, R1, R2, P1, P2, 
                                            chess_points_l[i], chess_points_b[i])[0][0])
    chess_corner = np.array(chess_corner)
    d = np.sqrt(np.sum((chess_corner[0] - chess_corner[1])**2))
    print(f"Distance between two pint: {d}, {d} = 1cm")
    chess_corner = chess_corner/d; point3d_in = point3d_in/d; point3d_th = point3d_th/d

    transformation_matrix = get_transformation_matrix(chess_corner)
    chess_corner_trans = [] # convert origin to left down chessboard 
    for i in range(len(chess_corner)):# rotate data points
        chess_corner_trans.append(np.dot(transformation_matrix, chess_corner[i].reshape(3,1)).reshape(1,3)[0])
    index_trans, thumb_trans = [], []
    for i in range(len(point3d_in)):
        index_trans.append(np.dot(transformation_matrix, point3d_in[i].reshape(3,1)).reshape(1,3)[0])
        # thumb_trans.append(np.dot(transformation_matrix, point3d_th[i].reshape(3,1)).reshape(1,3)[0])

    origin_point = chess_corner_trans[40] # translate data points
    chess_corner_trans = np.array(chess_corner_trans) - origin_point
    index_trans = np.array(index_trans) - origin_point
    # thumb_trans = np.array(thumb_trans) - origin_point

    df = pd.DataFrame()
    df["time"] = dt
    df["index_x"] = index_trans[:,0]; df["index_y"] = index_trans[:,1]; df["index_z"] = index_trans[:,2]
    # df["thumb_x"] = thumb_trans[:,0]; df["thumb_y"] = thumb_trans[:,1]; df["thumb_z"] = thumb_trans[:,2]
    # df["dis"] = np.sum(np.sqrt((index_trans - thumb_trans)**2), axis = 1)
    df.to_csv("../Result/3D_result.csv", index=False)