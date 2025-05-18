
""" 
Camera Calibration
"""

import numpy as np
import cv2
import glob
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calibrate_camera(chessboard_size=(9, 6), square_size=0.025):
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in the image plane
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images = glob.glob('images/left*.jpg')
    successful_images = 0

    for img_file in images:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_images += 1
            print(f"Chessboard found in {img_file}")
        else:
            print(f"Chessboard not found in {img_file}")

    if successful_images == 0:
        print("No valid chessboard patterns detected. Calibration failed.")
        return None, None

    img_size = gray.shape[::-1]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    return camera_matrix, dist_coeffs

def undistort_multiple_images(camera_matrix, dist_coeffs, pattern="./images/obj*.jpg", save_dir="./undistorted_results"):
    os.makedirs(save_dir, exist_ok=True)
    images = glob.glob(pattern)
    print(f"Found {len(images)} distorted object images to undistort.")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipped unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        base = os.path.basename(img_path)
        output_path = os.path.join(save_dir, f"undistorted_{base}")
        cv2.imwrite(output_path, dst)
        print(f"Saved: {output_path}")

def match_features_n(images, max_features=500):
    orb = cv2.ORB_create(max_features)
    keypoints = []
    descriptors = []
    for img in images:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_dict = {}
    n = len(images)
    for i in range(n):
        for j in range(i+1, n):
            raw = bf.match(descriptors[i], descriptors[j])
            raw = sorted(raw, key=lambda m: m.distance)
                        # Visualize the matches for debugging 
            if len(raw) > 10:  # only if there are enough matches
                img_matches = cv2.drawMatches(images[i], keypoints[i], images[j], keypoints[j], raw[:50], None, flags=2)
                match_vis_name = f"match_{i}_{j}.jpg"
                cv2.imwrite(match_vis_name, img_matches)
                print(f"Saved visual match between image {i} and {j} to {match_vis_name}")

            pts_i = np.array([keypoints[i][m.queryIdx].pt for m in raw], dtype=np.float32)
            pts_j = np.array([keypoints[j][m.trainIdx].pt for m in raw], dtype=np.float32)
            matches_dict[(i, j)] = (pts_i, pts_j, raw)
            print(f"Matched {len(raw)} features between image {i} and {j}.")
    return matches_dict

def reconstruct_pair(pts_i, pts_j, camera_matrix):
    # Normalize pixel coordinates using the camera intrinsics
    pts_i_norm = cv2.undistortPoints(np.expand_dims(pts_i, axis=1), camera_matrix, None).reshape(-1,2)
    pts_j_norm = cv2.undistortPoints(np.expand_dims(pts_j, axis=1), camera_matrix, None).reshape(-1,2)

    E, mask = cv2.findEssentialMat(pts_i_norm, pts_j_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts_i_norm, pts_j_norm)
    

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    pts4d = cv2.triangulatePoints(P1, P2, pts_i_norm.T, pts_j_norm.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # Keep only points in front of the camera (positive Z)
    pts3d = pts3d[pts3d[:, 2] > 0]

    return pts3d
def clean_point_cloud(points3d, distance_threshold=0.02, num_neighbors=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)
    print(f"Cleaned point cloud: {len(inlier_cloud.points)} points remaining after filtering")
    return np.asarray(inlier_cloud.points)

def reconstruct_mesh_poisson(points3d, depth=9):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth)
    mesh = mesh.filter_smooth_simple(number_of_iterations=10)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    print(f"Mesh reconstructed with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh

def save_mesh(mesh, filename='reconstructed_mesh.ply'):
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Mesh saved to {filename}")

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries([mesh])

def main():
    mtx, dist = calibrate_camera()
    if mtx is None or dist is None:
        print("Échec de la calibration : aucun damier détecté.")
        return

    undistort_multiple_images(mtx, dist)
    undistorted_files = sorted(glob.glob("./undistorted_results/undistorted_*.jpg"))
    if len(undistorted_files) < 2:
        print("Il faut au moins deux images corrigées pour faire du matching.")
        return
    print(f"Found {len(undistorted_files)} undistorted images: {undistorted_files}")

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

    matches_dict = match_features_n(imgs, max_features=500)
    all_pts = []
    for (i, j), (pts_i, pts_j, raw_matches) in matches_dict.items():
        print(f"Images {i} ↔ {j} : {len(raw_matches)} correspondances")
        if len(raw_matches) < 8:
            continue
        pts3d = reconstruct_pair(pts_i, pts_j, mtx)
        print(f"→ {pts3d.shape[0]} points 3D triangulated for pair ({i},{j})")
        all_pts.append(pts3d)

    if all_pts:
        cloud = np.vstack(all_pts)
        np.savetxt('points3d_all.txt', cloud, fmt='%.6f')
        print(f"Saved raw 3D point cloud: {cloud.shape[0]} points in points3d_all.txt")

        # ─── apply the same filter here ───
        raw_filtered = cloud[ (cloud[:, 0] < 100)]
        print(f"Plotting {raw_filtered.shape[0]} filtered raw points")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(raw_filtered[:, 0], raw_filtered[:, 1], raw_filtered[:, 2], s=1)
        ax.set_title("Filtered Raw 3D Point Cloud (X<100)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.show()
    else:
        print("Aucun point 3D à sauvegarder.")
        return

    pts = np.loadtxt('points3d_all.txt')
    cleaned_pts = clean_point_cloud(pts)
    mesh = reconstruct_mesh_poisson(cleaned_pts)
    save_mesh(mesh)
    visualize_mesh(mesh)

    # ─── same filter again on cleaned pts ───
    pt_copy = cleaned_pts[ (cleaned_pts[:, 0] < 100)]
    print(f"Plotting {pt_copy.shape[0]} filtered cleaned points")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pt_copy[:, 0], pt_copy[:, 1], pt_copy[:, 2], s=1)
    ax.set_title("Filtered Cleaned 3D Point Cloud (X<100)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.show()

if __name__ == "__main__":
    main()
