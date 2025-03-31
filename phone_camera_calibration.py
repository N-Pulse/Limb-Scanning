
""" 
Camera Calibration
"""

import numpy as np

import cv2
import glob
import os



def calibrate_camera(chessboard_size=(9, 6), square_size=0.025):

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images = glob.glob('./images/calibration*.jpg')
   # print(f"Found {len(images)} calibration images.")

    successful_images = 0

    for img_file in images:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)


        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_images += 1
        else:
            continue

    if successful_images == 0:
        return None, None

    img_size = gray.shape[::-1]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    """
        print(f"Calibration complete. Error: {ret}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs}")
        """


    return camera_matrix, dist_coeffs


def undistort_multiple_images(camera_matrix, dist_coeffs, pattern="./images/distorted*.jpg", save_dir="./undistorted_results"):
    os.makedirs(save_dir, exist_ok=True)
    images = glob.glob(pattern)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f" Skipped unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)

        ## might be optional(region of interest)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        base = os.path.basename(img_path)
        output_path = os.path.join(save_dir, f"undistorted_{base}")
        cv2.imwrite(output_path, dst)
        print(f" Saved: {output_path}")

def main():
    mtx, dist = calibrate_camera()
    if mtx is not None and dist is not None:
        undistort_multiple_images(mtx, dist)