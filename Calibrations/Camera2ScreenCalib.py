from __future__ import print_function

import sys, os, math, glob, argparse
import numpy as np
import cv2
from cv2 import aruco
from checkerboard import detect_checkerboard
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

PNGPattern = "*.png"

## Setup

# Checker Pattern(horizontal square count -1, vertical square count -1) project on display
CHECKERPATTERN_SIZE = (23, 7)
# Display dimensions
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
# Checker Pattern coordination in pixel unit
CHECKER_FILE = "Projections/8_24_checker.npz"
# Aruco marker on the mirror dimensions (m)
ARUCO_MARKER_LENGTH = 0.04
#ARUCO_BOARD_HEIGHT = 0.2884
#ARUCO_BOARD_WIDTH = 0.5932
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

def readFileList(imgFolder, ImgPattern=PNGPattern):
    imgFileList = glob.glob(os.path.join(imgFolder, ImgPattern))
    imgFileList.sort()
    return imgFileList

def detectChecker(img, patternSize=CHECKERPATTERN_SIZE, debug=True):
    # print(len(img.shape))
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE)
    corners_refine = corners
    #print(ret)
 
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refine = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

        if debug:
            cv2.drawChessboardCorners(img, patternSize, corners_refine, ret)
            width = 960
            height = int(img.shape[0] * 960 / img.shape[1])
            smallimg = cv2.resize(img, (width, height))
            # fig = plt.figure(figsize=(15,10))
            # plt.figure()
            # fig.imshow(smallimg)
            # print("image of checkerboard!!!!!!!!!!!!!!!!!!")
            # plt.show()
            # fig.show()

    return ret, corners_refine
def readCheckerObjPoint(fname):
    data = np.load(fname)
    objp = data["objp"]

    return objp
def arucoBoard(m=ARUCO_MARKER_LENGTH, h= 0.2884, w= 0.5932):

    # create objPoints for calibration target
    h0 = (0 - h / 2)
    hm = (m - h / 2)
    h1 = (((h - m) / 2) - h / 2)
    h2 = (((h + m) / 2) - h / 2)
    h3 = ((h - m) - h / 2)
    h4 = (h - h / 2)
    w0 = (0 - w / 2)
    wm = (m - w / 2)
    w1 = (((w - m) / 2) - w / 2)
    w2 = (((w + m) / 2) - w / 2)
    w3 = ((w - m) - w / 2)
    w4 = (w - w / 2)


    objPoints = []
    objPoints.append(np.array([[w0, h0, 0], [wm, h0, 0], [wm, hm, 0], [w0, hm, 0]], dtype=np.float32))  # 0
    objPoints.append(np.array([[w0, h1, 0], [wm, h1, 0], [wm, h2, 0], [w0, h2, 0]], dtype=np.float32))  # 1
    objPoints.append(np.array([[w0, h3, 0], [wm, h3, 0], [wm, h4, 0], [w0, h4, 0]], dtype=np.float32))  # 2
    objPoints.append(np.array([[w1, h3, 0], [w2, h3, 0], [w2, h4, 0], [w1, h4, 0]], dtype=np.float32))  # 3
    objPoints.append(np.array([[w3, h3, 0], [w4, h3, 0], [w4, h4, 0], [w3, h4, 0]], dtype=np.float32))  # 4
    objPoints.append(np.array([[w3, h1, 0], [w4, h1, 0], [w4, h2, 0], [w3, h2, 0]], dtype=np.float32))  # 5
    objPoints.append(np.array([[w3, h0, 0], [w4, h0, 0], [w4, hm, 0], [w3, hm, 0]], dtype=np.float32))  # 6
    objPoints.append(np.array([[w1, h0, 0], [w2, h0, 0], [w2, hm, 0], [w1, hm, 0]], dtype=np.float32))  # 7

    ids = np.linspace(0, 7, 8).astype(np.int32)[:, None]
    arucoCornerBoard = aruco.Board_create(objPoints, aruco_dict, ids)

    return arucoCornerBoard, objPoints

def detectAruco(img, debug=True):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = img

    parameters = aruco.DetectorParameters_create()
    #corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if debug:
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        width = 960
        height = int(img.shape[0] * 960 / img.shape[1])
        smallimg = cv2.resize(frame_markers, (width, height))
        plt.figure(figsize=(15,10))
        plt.imshow(smallimg)
        plt.show()
        # print("image of Aruco!!!!!!!!!!!!!!!!!!")

    return corners, ids

def postEst(corners, ids, camMat, distCoeffs, markerLength=ARUCO_MARKER_LENGTH, arucoBoardHeight= 0.2884, arucoBoardWidth= 0.5932):

    arucoCornerBoard, _ = arucoBoard(markerLength, arucoBoardHeight, arucoBoardWidth)

    rvec = []
    tvec = []
    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, arucoCornerBoard, camMat, distCoeffs, None, None)

    return rvec, tvec

def reProjAruco(img, camMat, distCoeffs, rvec, tvec, cornersAruco, markerLength=ARUCO_MARKER_LENGTH, arucoBoardHeight= 0.2884, arucoBoardWidth= 0.5932):
    print("reProjAruco")
    _, objPoints = arucoBoard(markerLength, arucoBoardHeight, arucoBoardWidth) #yunhao

    ids = np.linspace(0, 7, 8).astype(np.int32)[:, None]
    corners_reproj = []
    for i in range(len(objPoints)):
        imgPoints, _ = cv2.projectPoints(np.array(objPoints[i]), rvec, tvec, camMat, distCoeffs)
        corners_reproj.append(imgPoints)

    frame_markers = aruco.drawDetectedMarkers(img.copy(), corners_reproj, ids)
    # cv2.imwrite("./reproejct_markers.png", frame_markers)
    fig = plt.figure(figsize=(15,10))
    plt.imshow(frame_markers)
    imgPath = os.path.join(os.path.join(os.path.join(os.getcwd(), 'CalibrationImages'), 'Geometric'), 'geoCalibResults')
    fig.savefig(imgPath + '/reproejctMarkers' + '.png')
    plt.show()

def householderTransform(n, d):
    I3 = np.identity(3, dtype=np.float32)
    e = np.array([0, 0, 1])
    p1 = I3 - 2 * np.outer(n, n)
    p2 = I3 - 2 * np.outer(e, e)
    p3 = 2 * d * n

    return p1, p2, p3

def invTransformation(R, t):
    Rinv = R.T
    Tinv = -(Rinv@t)

    return Rinv, Tinv

def calib(imgPath, camMtx, dist, half_length, half_height, displayScaleFactor):
    ARUCO_BOARD_HEIGHT = half_height * 2 
    ARUCO_BOARD_WIDTH = half_length * 2 
    CHECKSQUARE_SIZE = displayScaleFactor

    objP_pixel = np.ceil(readCheckerObjPoint(CHECKER_FILE))
    objP_pixel[:, 2] = 0
    objP = np.array(objP_pixel)
    for i in range(CHECKERPATTERN_SIZE[1]):
        for j in range(math.floor(CHECKERPATTERN_SIZE[0]/2)):
            tmp = objP[CHECKERPATTERN_SIZE[0] * i + j, 0]
            objP[CHECKERPATTERN_SIZE[0] * i + j, 0] = objP[CHECKERPATTERN_SIZE[0] * i + CHECKERPATTERN_SIZE[0] - j - 1, 0]
            objP[CHECKERPATTERN_SIZE[0] * i + CHECKERPATTERN_SIZE[0] - j - 1, 0] = tmp
    objP[:, 0] -= (SCREEN_WIDTH / 2 - 1)
    objP[:, 1] -= (SCREEN_HEIGHT / 2 - 1)

    objP *= CHECKSQUARE_SIZE
    

    rtA = []
    rB = []
    tB = []
    rC2Ss = []
    tC2Ss = []

    # define valid image
    validImg = -1
    #for i in trange(len(imgFileList), desc="Images"):
    for i in range(1):
        
        #img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(imgPath)
        # print(img.shape)
        # Yunhao
        # img = (img/65535*255).astype(np.uint8)
        #print(len(img.shape))
        # plt.imshow(img)
        # plt.colorbar()
        # Aruco marker for Mirror position
        cornersAruco, ids = detectAruco(img, debug=False)
        if cornersAruco is None and ids is None and len(cornersAruco) <= 3:
            continue

        # Checker for Display
        # print(len(img.shape))
        ret, cornersChecker = detectChecker(img, CHECKERPATTERN_SIZE, debug=True)
        # print(ret)
        # print(cornersChecker)
        if not ret:
            print("no Checker!!!")
            continue
        
        #for a valid image, aruco and checker must be both detected
        validImg += 1

        # Calibrate Mirror Pose with Aruco

        rvecMirror, tvecMirror = postEst(cornersAruco, ids, camMtx, dist, ARUCO_MARKER_LENGTH,
                                         ARUCO_BOARD_HEIGHT, ARUCO_BOARD_WIDTH)
        img_axis = aruco.drawAxis(img, camMtx, dist, rvecMirror, tvecMirror, ARUCO_MARKER_LENGTH)
        width = 960
        height = int(img_axis.shape[0] * 960 / img_axis.shape[1])
        smallimg = cv2.resize(img_axis, (width, height))
        # plt.figure(figsize=(15,10))
        # plt.imshow(smallimg)
        # plt.show()


        ## Reproejct Camera Extrinsic
        reProjAruco(img, camMtx, dist, rvecMirror, tvecMirror, cornersAruco)

        rMatMirror, _ = cv2.Rodrigues(rvecMirror)  # rotation vector to rotation matrix
        normalMirror = rMatMirror[:, 2]

        rC2W, tC2W = invTransformation(rMatMirror, tvecMirror)
        dW2C = abs(np.dot(normalMirror, tvecMirror))

        # Householder transformation
        p1, p2, p3 = householderTransform(normalMirror, dW2C)

        # Calibrate virtual to Camera with Checker
        rpe, rvecVirtual, tvecVirtual = cv2.solvePnP(objP, cornersChecker, camMtx, dist, flags=cv2.SOLVEPNP_IPPE)  # cv2.SOLVEPNP_IPPE for 4 point solution #cv2.SOLVEPNP_ITERATIVE
                                                                      #iterationsCount=200, reprojectionError=8.0,

        rvecVirtual, tvecVirtual = cv2.solvePnPRefineLM(objP, cornersChecker, camMtx, dist, rvecVirtual,
                                                        tvecVirtual)

        proj, jac = cv2.projectPoints(objP, rvecVirtual, tvecVirtual, camMtx, dist)
        img_rep = img

        cv2.drawChessboardCorners(img_rep, CHECKERPATTERN_SIZE, proj, True)
        width = 960
        height = int(img_rep.shape[0] * 960 / img_rep.shape[1])
        smallimg = cv2.resize(img_rep, (width, height))
        fig = plt.figure(figsize=(15,10))
        plt.imshow(smallimg)
        imgPath = os.path.join(os.path.join(os.path.join(os.getcwd(), 'CalibrationImages'), 'Geometric'), 'geoCalibResults')
        fig.savefig(imgPath + '/chesssBoard' + '.png')
        plt.show()


        rMatVirtual, _ = cv2.Rodrigues(rvecVirtual)  # rotation vector to rotation matrix

        print(tvecVirtual)
        if validImg == 0:
            rtA = p1
            rB = np.matmul(rMatVirtual, p2)
            tB = np.squeeze(tvecVirtual) + p3
        else:
            rtA = np.concatenate((rtA, p1))
            rB = np.concatenate((rB, np.matmul(rMatVirtual, p2)))
            tB = np.concatenate((tB, np.squeeze(tvecVirtual) + p3))

        rS2C = p1 @ rMatVirtual
        tS2C = p1 @ np.squeeze(tvecVirtual) + p3

        rC2S, tC2S = invTransformation(rS2C, tS2C)
        print("rC2S:", rC2S)
        print("tC2S:", tC2S)
        rC2Ss.append(rC2S)
        tC2Ss.append(tC2S)

    return rC2Ss, tC2Ss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path of images directory")
    parser.add_argument("-c", "--calibration", type=str, help="Path to intrinsic calibration npz")
    args = parser.parse_args()

    if args.directory is not None:
    	imgFolder = os.path.normpath(args.directory)
    if args.calibration is not None:
    	cameraCalibPath = os.path.normpath(args.calibration)

    calib(imgFolder, cameraCalibPath)