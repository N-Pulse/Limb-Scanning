
""" 
Camera Calibration
"""

import numpy as np

import cv2
import glob
import os

  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def calibrate_camera(chessboard_size=(9, 6), square_size=0.025):

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images = glob.glob('images/left*.jpg')
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


def undistort_multiple_images(camera_matrix, dist_coeffs, pattern="./images/dinoSR*.png", save_dir="./undistorted_results"):
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

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        base = os.path.basename(img_path)
        output_path = os.path.join(save_dir, f"undistorted_{base}")
        cv2.imwrite(output_path, dst)
        print(f" Saved: {output_path}")
        
def match_features_n(images, max_features=500):
    """
    Détecte et met en correspondance les features ORB pour N images.

    Args:
        images (list of np.ndarray): liste d’images (BGR ou niveaux de gris).
        max_features (int): nombre max de keypoints ORB par image.

    Returns:
        matches_dict (dict): clés (i,j) pour chaque paire d’indices d’images,
                             valeurs (pts_i, pts_j, matches) où
                             - pts_i, pts_j sont des arrays (Mi×2) et (Mi×2) des coords float32,
                             - matches est la liste des cv2.DMatch triés par distance.
    """
    # 1) Détection ORB et description
    orb = cv2.ORB_create(max_features)
    keypoints = []
    descriptors = []
    for img in images:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    # 2) Initialisation du matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 3) Correspondances pour chaque paire (i,j)
    matches_dict = {}
    n = len(images)
    for i in range(n):
        for j in range(i+1, n):
            raw = bf.match(descriptors[i], descriptors[j])
            raw = sorted(raw, key=lambda m: m.distance)

            # Extraction des points appariés
            pts_i = np.array([ keypoints[i][m.queryIdx].pt for m in raw ], dtype=np.float32)
            pts_j = np.array([ keypoints[j][m.trainIdx].pt  for m in raw ], dtype=np.float32)

            matches_dict[(i, j)] = (pts_i, pts_j, raw)

    return matches_dict

def reconstruct_pair(pts_i, pts_j, camera_matrix):
    E, mask = cv2.findEssentialMat(pts_i, pts_j, camera_matrix,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts_i, pts_j, camera_matrix)
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = camera_matrix @ np.hstack((R, t))
    pts4d = cv2.triangulatePoints(P1, P2, pts_i.T, pts_j.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d

def main():
    # 1) Calibration
    mtx, dist = calibrate_camera()
    if mtx is None or dist is None:
        print("Échec de la calibration : aucun damier détecté.")
        return

    # 2) Undistortion de toutes les images déformées
    undistort_multiple_images(mtx, dist)

    # 3) Chargement des images corrigées
    undistorted_files = sorted(glob.glob("./undistorted_results/undistorted_*.png"))
    if len(undistorted_files) < 2:
        print("Il faut au moins deux images corrigées pour faire du matching.")
        return

    imgs = []
    for path in undistorted_files:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Impossible de lire {path}, skip.")
            continue
        imgs.append(img)

    if len(imgs) < 2:
        print("Pas assez d'images valides.")
        return

    # 4) Mise en correspondance des features pour N images
    matches_dict = match_features_n(imgs, max_features=500)

    # 5) Traitement des correspondances et triangulation
    for (i, j), (pts_i, pts_j, raw_matches) in matches_dict.items():
        print(f"Images {i} ↔ {j} : {len(raw_matches)} correspondances")
        if len(raw_matches) >= 8:
            pts3d = reconstruct_pair(pts_i, pts_j, mtx)
            print(f"  → {pts3d.shape[0]} points 3D triangulés pour la paire ({i},{j})")
    all_pts = []
    for (i, j), (pts_i, pts_j, raw_matches) in matches_dict.items():
        if len(raw_matches) < 8:
            continue
        pts3d = reconstruct_pair(pts_i, pts_j, mtx)
        all_pts.append(pts3d)

    if all_pts:
        cloud = np.vstack(all_pts)
        np.savetxt('points3d_all.txt', cloud, fmt='%.6f')
        print(f"Nuage global sauvegardé: {cloud.shape[0]} points dans points3d_all.txt")
    else:
        print("Aucun point 3D à sauvegarder.")
    # Chargez les points
    pts = np.loadtxt('points3d_all.txt')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    main()



