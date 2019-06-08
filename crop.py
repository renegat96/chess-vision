"""
MODIFIED CODE FROM DAYLEN YANG
https://github.com/daylen/chess-id
UNDER MIT LICENSE
"""
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
from functools import partial
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

""" LOGIC """

def auto_canny(image, sigma=3.9):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower, upper)
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def hor_vert_lines(lines):
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v

def intersections(h, v):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)

def cluster(points, max_dist=50):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = list(map(lambda arr: (np.mean(np.array(arr)[:,0]),
    np.mean(np.array(arr)[:,1])), clusters))
    return clusters

def closest_point(points, loc):
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(list(map(partial(spatial.distance.euclidean, loc), points)))
    return points[dists.argmin()]

def find_corners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closest_point(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))
    
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25 # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closest_point(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closest_point(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner), np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners

def four_point_transform(img, points, square_length=1816):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))

def find_board(fname):
    """
    Given a filename, returns the board image.
    """
    img = cv2.imread(fname, 1)
    if img is None:
        print('no image')
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    
    # Canny edge detection
    edges = auto_canny(gray)
    cv2.imwrite('edges.jpg', edges)
    #cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    #cv2.imshow('edges', edges)
    #cv2.resizeWindow('edges',600,600)
    #cv2.waitKey()
    print(np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) )
    if np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) > 0.035:
        print('too many edges')
        return None

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        print('no lines')
        return None

    lines = np.reshape(lines, (-1, 2))

    # Compute intersection points
    h, v = hor_vert_lines(lines)
    if len(h) < 9 or len(v) < 9:
        print('too few lines')
        return None
    points = intersections(h, v)

    # Cluster intersection points
    points = cluster(points)
    #for point in points:
    #    print(point)
    #    cv2.circle(img, point, 20, (0, 0, 255), 4)
    #cv2.namedWindow('board', cv2.WINDOW_NORMAL)
    #cv2.imshow('board', img)
    #cv2.resizeWindow('board',600,600)
    #cv2.waitKey()

    # Find corners
    img_shape = np.shape(img)
    points = find_corners(points, (img_shape[1], img_shape[0]))

    # Perspective transform
    new_img = four_point_transform(img, points)

    cv2.imwrite('transformed.jpg', new_img)
    return new_img

def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len])
    return arr

def shrink_blanks(fen):
    if '_' not in fen:
        return fen
    new_fen = ''
    blanks = 0
    for char in fen:
        if char == '_':
            blanks += 1
        else:
            if blanks != 0:
                new_fen += str(blanks)
                blanks = 0
            new_fen += char
    if blanks != 0:
        new_fen += str(blanks)
    return new_fen

def get_fen(arr):
    fen = ''
    for sq in arr:
        if sq == 'empty':
            fen += '_'
        elif sq[0] == 'b':
            fen += sq[1]
        else:
            fen += str(sq[1]).upper()
    fens = [fen[i:i+8] for i in range(0, 64, 8)]
    fens = list(map(shrink_blanks, fens))
    fen = '/'.join(fens)
    return fen

if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=False)
    boardFileName = "test.jpg"
    args = sys.argv
    if len(args) > 1:
        boardFileName = args[-1]
    board = find_board(boardFileName)
    if board is None:
        print("no board could be processed")
    else:
        squares = split_board(board)
        squares = [preprocess_input(cv2.resize(sq, (224, 224))) for sq in squares]
        squares = np.array(squares)
        print(squares.shape)
        model = load_model('best_model.h5')
        y = model.predict(squares)
        y_pred = np.argmax(y,axis=1)
        y_pred = [categories[i] for i in y_pred]
        print(y_pred)
        print(get_fen(y_pred))
        #cv2.namedWindow('board', cv2.WINDOW_NORMAL)
        #cv2.imshow('board', board)
        #cv2.resizeWindow('board',600,600)
        #cv2.waitKey()

