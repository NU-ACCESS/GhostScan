import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import pickle

def intrinsicWithChessboard(checkerH, checkerV, size, imgDistortFolderPath):
    checkerH = checkerH
    checkerV = checkerV
    size = size
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerH*checkerV,3), np.float32)
    # measurement meter
    # square length - size 
    # 26.4 mm
    objp[:,:2] = np.mgrid[0:checkerV*size:size, 0:checkerH*size:size].T.reshape(-1,2)

    #objp[:,:2] = np.mgrid[0:8:1, 0:6:1].T.reshape(-1,2)
    # print(objp[:,:2])
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    # checkerboard
    images = glob.glob('CalibrationImages/intrinsicChessboard/*.png')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (checkerV,checkerH), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (checkerV,checkerH), corners, ret)
            write_name = 'CalibrationImages/intrinsicChessboard/intrisic result/corners_found'+str(idx)+'.png'
            cv2.imwrite(write_name, img)
    #         cv2.imshow('img', img)
            cv2.waitKey(10)

    cv2.destroyAllWindows()
    
    imgDistortFolderPath = imgDistortFolderPath
    # imgDistortFolderPath = os.path.join(os.path.join(os.getcwd(), 'CapturedImages'), 'sequenceImages')
    imgFileList = readFileList(imgDistortFolderPath)
    index = 0;

    for i in imgFileList:

        print("reading %s" % i)
        img = cv2.imread(i)
    #     img = cv2.imread('intrinsic/test.png')
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


        dst = cv2.undistort(img, mtx, dist, None, mtx)
    #     imgDistortFilename = os.path.join(imgDistortFolder,'undistort', str(img_num) + '.png')
    #   cv2.imwrite(imgDistortFilename, dst)
        
        pathName = imgDistortFolderPath
        write_name = pathName + '/undistort/'+str(index)+'.png'
        cv2.imwrite(write_name, dst)

        index += 1
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )
    #   dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)

        if len(imgFileList) == 1:
            return ret, mtx, dist, rvecs, tvecs
    

def readFileList(imgFolder, ImgPattern="*.png"):
    imgFileList = glob.glob(os.path.join(imgFolder, ImgPattern))
    # self.imgFileList = os.listdir(self.imgFolder)
    # self.imgFileList.remove('.DS_Store') # remove system database log
    #imgFileList.sort(key=lambda x:int(x[len(imgFolder) + 1: -4]))
    if len(imgFileList) == 1:
        imgFileList.sort()
    else:
        imgFileList.sort()
        imgFileSubList = imgFileList[0:len(imgFileList) - 1]
        imgFileSubList.sort(key=lambda x:int(x[len(imgFolder) + 1: -4]))
        imgFileList[0:len(imgFileList) - 1] = imgFileSubList
        #print(imgFileSubList)
    #print(imgFileList)
    return imgFileList

