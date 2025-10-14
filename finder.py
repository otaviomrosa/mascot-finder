import numpy as np
import math
import cv2

"""
MATCHING:

    OpenCV's sift function was used to find keypoints in both the reference and in the test image.
    Matching was done using euclidean distance and Lowe's ratio test, and then pairs sharing the same test image keypoint were discarded (all but one).
"""

def find_matches(reference, image):

    # initialize sift
    sift = cv2.SIFT.create(nfeatures=3000)

    # get points and descriptor for reference image
    ref_kp, ref_desc = sift.detectAndCompute(reference, None)

    # get points and descriptor for test image
    img_kp, img_desc = sift.detectAndCompute(image, None)

    initial_pairs = []

    for r_kp, r_desc in zip(ref_kp, ref_desc):

        distances = np.linalg.norm(r_desc - img_desc, axis=1) # using broadcasting to subtract the current r_desc from all img_desc values
                                                              # numpy "stretches" r_desc x times so that it has the same size as img_desc, and then performs the subtraction

        indices = np.argsort(distances) # get index of all distances in sorted order (we do this because we need the same index in both arrays
        match_idx = indices[0]          # lowest distance
        sec_match_idx = indices[1]      # second-lowest distance (used below)

        if distances[match_idx] / distances[sec_match_idx] < 0.75: # Lowe's ratio test
            initial_pairs.append((r_kp, img_kp[match_idx]))

    unique_pairs = []
    used_ikp = set()

    # remove one of the pairs when two reference keypoints maps to the same test image keypoint
    for r_kp, i_kp in initial_pairs: # to improve, consider sorting initial_pairs in terms of distance between the two points to prioritize best matches (include third element in the tuple for the distance value)
           if i_kp not in used_ikp:
               unique_pairs.append((r_kp, i_kp))
               used_ikp.add(i_kp)

    return unique_pairs

""" 
RANSAC:

    Use pairs matched by find_matches as data for ransac. pick three random points n times (arbitrary) and find which transform
    is best for the most amount of reference points. Use this transform matrix for the next step.
"""

def ransac(pairs, num_iterations, threshold):

    max_inliers = 0
    best_transform = None

    for i in range(num_iterations):

        indices = np.random.choice(np.arange(len(pairs)), size=3, replace=False) # you could also use the random library, but I wanted to learn the numpy equivalent.

        ref_pts = np.float32([pairs[i][0].pt for i in indices]) # get actual coordinates through list comprehension. note
        img_pts = np.float32([pairs[i][1].pt for i in indices])

        m = cv2.getAffineTransform(ref_pts, img_pts)

        inliers = 0

        for ref_kp, img_kp in pairs: # if this is too slow consider finding a way to process all transformations at once like we did with the matching

            ref_m = np.float32([ref_kp.pt]).reshape(-1, 1, 2) # transform the reference keypoint into the format taken by cv2.transform
            transformed_kp = cv2.transform(ref_m, m)[0][0] # this returns a [[[x, y]]] object, we need to get the innermost list for the Euclidean function.

            if np.linalg.norm(np.array(img_kp.pt) - transformed_kp) < threshold:
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            best_transform = m

    return best_transform

"""
BOX INFO:

    Find the corners of the reference picture, transform them using the matrix found through ransac, use simple algebra
    to find the center coordinates, height, and the arctan function to find the angle. Normalize the angle to the specifications (0 is upright)
"""

def box_info(reference, matrix):

    info = []
    h, w = reference.shape[:2] #no need to get channel
    corners = np.float32([[w, h], [0, h], [w ,0], [0,0]]).reshape(-1, 1, 2) # order is top right, top left, bottom right, bottom left
    trans_corners = cv2.transform(corners, matrix).reshape(4, 2) # transform the output into a normal array

    x_center = (trans_corners[0][0] + trans_corners[1][0] + trans_corners[2][0] + trans_corners[3][0])/4 # simple average
    y_center = (trans_corners[0][1] + trans_corners[1][1] + trans_corners[2][1] + trans_corners[3][1])/4

    info.append(x_center)
    info.append(y_center)

    height = np.linalg.norm(trans_corners[0] - trans_corners[2]) # distance between top and bottom of the same side

    info.append(height)

    rad = math.atan2(trans_corners[3][1] - trans_corners[1][1], trans_corners[3][0] - trans_corners[1][0])
    deg = np.rad2deg(rad)
    angle = int((deg + 90) % 360) # normalize and take the modulus to always get a value between 0 and 360

    info.append(angle)

    return info

"""
MATCH:
    
     This function identifies the intersection over union (IoU) between an ideal box and the box found by our algorithm
     as a value between 0 and 1. This is only used for testing
"""

def match(img_shape, cx, cy, h, a, gt_cx, gt_cy, gt_h, gt_a):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    sina = np.sin(np.pi * a / 180.0)
    cosa = np.cos(np.pi * a / 180.0)

    vx, vy = -sina * h / 2.0, cosa * h / 2.0
    hx, hy = cosa * 0.6 * h / 2.0, sina * 0.6 * h / 2.0

    pts = [[cx - vx - hx, cy - vy - hy], [cx - vx + hx, cy - vy + hy], [cx + vx + hx, cy + vy + hy],
           [cx + vx - hx, cy + vy - hy]]

    for y in range(0, img_shape[0], 3):
        for x in range(0, img_shape[1], 3):
            flag = True
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i + 1) % 4]
                if (p2[1] - p1[1]) * x - (p2[0] - p1[0]) * y + p2[0] * p1[1] - p2[1] * p1[0] > 0:
                    flag = False
            if flag:
                mask[y, x] = 1

    mask2 = np.zeros(img_shape[:2], dtype=np.uint8)

    sina = np.sin(np.pi * gt_a / 180.0)
    cosa = np.cos(np.pi * gt_a / 180.0)

    vx, vy = -sina * gt_h / 2.0, cosa * gt_h / 2.0
    hx, hy = cosa * 0.6 * gt_h / 2.0, sina * 0.6 * gt_h / 2.0

    pts = [[gt_cx - vx - hx, gt_cy - vy - hy], [gt_cx - vx + hx, gt_cy - vy + hy], [gt_cx + vx + hx, gt_cy + vy + hy],
           [gt_cx + vx - hx, gt_cy + vy - hy]]

    for y in range(0, img_shape[0], 3):
        for x in range(0, img_shape[1], 3):
            flag = True
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i + 1) % 4]
                if (p2[1] - p1[1]) * x - (p2[0] - p1[0]) * y + p2[0] * p1[1] - p2[1] * p1[0] > 0:
                    flag = False
            if flag:
                mask2[y, x] = 1

    iou = np.sum(mask * mask2) / np.sum(np.bitwise_or(mask, mask2))

    return iou

if __name__ == "__main__":

    ref = cv2.imread('./tests/reference.png', cv2.IMREAD_COLOR)
    if ref is None:
        print(f"Error: Could not read reference image")

    img = cv2.imread('./tests/1.png', cv2.IMREAD_COLOR) # change number (1-14) to test different pictures
    if img is None:
        print(f"Error: Could not read test image")


    matches = find_matches(ref, img)
    transform = ransac(matches, 3000, 3)

    X, Y, H, A = box_info(ref, transform)
    out_str = f"{int(X)} {int(Y)} {int(H)} {int(A)}"

    with open("output.txt", 'w') as f:
        f.write(out_str)

    print(out_str)
