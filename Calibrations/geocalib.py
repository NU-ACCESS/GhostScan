import sys, glob, os
import math
import argparse
import numpy as np
import cv2
from cv2 import aruco
from Calibrations.Camera2ScreenCalib import calib
# from Camera2ScreenCalib import calib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def geoCalib(imgFile, imgUndistortFile, half_length, half_height, cameraMatrix, disCoeffs, ret, displayScaleFactor):

    # calculate cam2screen
    rc2s, tc2s = calib(imgFile, cameraMatrix, disCoeffs, half_length, half_height, displayScaleFactor)
    tc2s = tc2s[0]
    #print("new rc2s is ", rc2s)
    #print("new tc2s is ", tc2s)
    rcamera_dis = rc2s
    tcamera_dis = tc2s
    # additional section: Make sure that the shape of rC2S and tC2S is correct
    rcamera_dis = rcamera_dis[0]
    tcamera_dis = np.reshape(tcamera_dis, (3,))
    #print("rc2s is ",rcamera_dis)
    #print("tc2s is ",tcamera_dis)
    #print("intrinsic matrix is ",cameraMatrix)



    #print(intrinsic_mat.f.dist)
    allCorners = []
    allIds = []
    decimator = 0
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    #board = 
    img = cv2.imread(imgFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cornerRefine = cv2.aruco.CORNER_REFINE_CONTOUR
    [markerCorners,markerIDs, rejectedImgPoints] = cv2.aruco.detectMarkers(gray, dictionary, cameraMatrix=cameraMatrix, distCoeff=disCoeffs)



    board_corner = np.array([np.array([[-half_length, -half_height, 0], [-half_length + 0.04, -half_height, 0], [-half_length + 0.04, -half_height + 0.04, 0], [-half_length, -half_height + 0.04, 0]]),
                    np.array([[half_length - 0.04, -half_height, 0], [half_length, -half_height, 0], [half_length, -half_height + 0.04, 0], [half_length - 0.04, -half_height + 0.04, 0]]),
                    np.array([[half_length - 0.04, half_height - 0.04, 0], [half_length, half_height - 0.04, 0], [half_length, half_height, 0], [half_length - 0.04, half_height, 0]]),
                    np.array([[-half_length, half_height - 0.04, 0], [-half_length + 0.04, half_height - 0.04, 0], [-half_length + 0.04, half_height, 0], [-half_length, half_height, 0]]),
                    np.array([[-0.02, -half_height, 0], [0.02, -half_height, 0], [0.02, -half_height + 0.04, 0], [-0.02, -half_height + 0.04, 0]]),
                    np.array([[-half_length, -0.02, 0], [-half_length + 0.04, -0.02, 0], [-half_length + 0.04, 0.02, 0], [-half_length, 0.02, 0]]),
                    np.array([[half_length - 0.04, -0.02, 0], [half_length, -0.02, 0], [half_length, 0.02, 0], [half_length - 0.04, 0.02, 0]]),
                    np.array([[-0.02, half_height - 0.04, 0], [0.02, half_height - 0.04, 0], [0.02, half_height, 0], [-0.02, half_height, 0]])], dtype=np.float32)

    board_id = np.array([[0], [6], [4], [2], [7], [1], [5], [3]], dtype=np.int32)
    board = cv2.aruco.Board_create(board_corner, dictionary, board_id)





    if len(markerCorners) > 0:
        allCorners.append(markerCorners)
        allIds.append(markerIDs)
        cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIDs, [0,200,0])



    #cv2.aruco.drawPlanarBoard(board, img.shape)
    rvecs,tvecs,_objpoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.04, cameraMatrix, disCoeffs, )
    r_mean = np.mean(rvecs,axis=0)
    t_mean = np.mean(tvecs,axis=0)

    #r_mirror = np.zeros((1,3))
    #t_mirror = np.zeros((1,3))
    r_mirror = None
    t_mirror = None
    ret, r_mirror, t_mirror, = cv2.aruco.estimatePoseBoard(markerCorners, markerIDs, board, cameraMatrix, disCoeffs, r_mirror, t_mirror)
    #print("r vec of mirror is ", r_mirror)
    #print("t vec of mirror is ", t_mirror)
    r_mirror_mat = np.zeros((3,3))
    r_mirror_mat = cv2.Rodrigues(r_mirror,dst= r_mirror_mat,jacobian=None)
    r_mirror_mat = r_mirror_mat[0]
    #print("r mirror mat is ", r_mirror_mat)
    img_axis = aruco.drawAxis(img, cameraMatrix, disCoeffs, r_mirror, t_mirror, 0.04)
    width = 960
    height = int(img.shape[0]*960/img.shape[1])
    smallimg = cv2.resize(img,(width,height))
    smallimg = cv2.cvtColor(smallimg, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(15,10))
    plt.imshow(smallimg)
    # print(str(decimator) + '!!!!!!!!!!!!')
    imgSavePath = os.path.join(os.path.join(os.path.join(os.getcwd(), 'CalibrationImages'), 'Geometric'), 'geoCalibResults')
    fig.savefig(imgSavePath + '/findAllAruco' + '.png')
    plt.show()
    decimator+=1
    imsize = img.shape
    #print("size of img is ", imsize)


    #rtcam: camera coordinate system to world coordinate system
    #rtMat: world coordinate system to camera coordinate system
    rtMat = np.hstack((r_mirror_mat, t_mirror))
    rtMat = np.vstack((rtMat, np.array([[0, 0, 0, 1]])))
    rt_cam = np.linalg.inv(rtMat)
    r_cam = rt_cam[0:3, 0:3]
    t_cam = rt_cam[0:3, 3]
    #print("rtMat is", rtMat)
    #print("rtcam in world coor is ", rt_cam)

    #rtdis: display coordinate system to world coordinate system
    #rtdis_inv: world coordinate system to display coordinate system
    t_camdis = np.reshape(tcamera_dis, (tcamera_dis.shape[0],1))
    rtMat_camdis = np.hstack((rcamera_dis, t_camdis))
    rtMat_camdis = np.vstack((rtMat_camdis, np.array([0,0,0,1])))
    #rtMat_discam = np.linalg.inv(rtMat_camdis)
    #rt_dis_inv = np.matmul(rtMat, rtMat_camdis)
    rt_dis_inv = np.matmul(rtMat_camdis, rtMat)
    rt_dis = np.linalg.inv(rt_dis_inv)
    r_dis = rt_dis[0:3, 0:3]
    t_dis = rt_dis[0:3,3]
    r_dis_inv = rt_dis_inv[0:3,0:3]
    t_dis_inv = rt_dis_inv[0:3,3]

    #rtMat_camdis: camera coordinate to display coordinate system
    #rtMat_discam: display coordinate to camera coordinate system
    rtMat_discam = np.linalg.inv(rtMat_camdis)


    #print("rtscreen in world coor is ", rt_dis)
    #print("mirror in screen coor is ",rt_dis_inv)
    #print("display to camera coor is ", rtMat_discam)
    #reshape t vector
    t_cam = np.reshape(t_cam, (t_cam.shape[0], 1))
    t_dis = np.reshape(t_dis, (t_dis.shape[0], 1))
    t_dis_inv = np.reshape(t_dis_inv, (t_dis_inv.shape[0], 1))



    # read undistorted image
    img_undistort = cv2.imread(imgUndistortFile)
    gray_undistort = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)
    img_undistort_size = gray_undistort.shape
    #print(img_undistort_size)


    #cam roor mat:store arrival vectors (vector from camera to the object) in camera coordinate system
    #cam world coor mat: store arrival vectors in world coordinate system
    #camera mirror intersect mat: store intersect points between arrival vectors and mirror in world coordinate system
    # reflec mat: store reflectance vectors (vector from object to screen) in world coordinate system 
    # mirror intersect mat trans:store intersect points on mirrors at display coordinate system
    # reflect mat trans: store reflectance vectors in display coordinate system
    # display intersect mat: store display intersect points (intersect points between reflectance vectors and display) on display coordinate system
    # display intersect mat trans: store display intersect points on world coordinate system
    img_coor_mat = np.zeros((3, img_undistort_size[0], img_undistort_size[1]))
    img_coor_mat_rs = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    cam_coor_mat = np.zeros((img_undistort_size[0], img_undistort_size[1], 3))
    cam_coor_mat_rs = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    cam_world_coor_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    cam_mirror_intersect_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    reflect_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    mirror_intersect_mat_trans = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    reflect_mat_trans = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    display_intersect_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
    display_intersect_mat_trans = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))

    #print(cam_coor_mat.shape)
    #print(np.linalg.inv(cameraMatrix))
    for i in range(img_undistort_size[0]):
        for j in range(img_undistort_size[1]):
            image_coor = np.array([j + 1, i + 1, 1])
            img_coor_mat[:, i, j] = image_coor

    img_coor_mat_rs = np.reshape(img_coor_mat, (3, img_undistort_size[0] * img_undistort_size[1]))
    #print(img_coor_mat_rs[:,1201])
    # calculate arrival vector
    cam_coor_mat_rs = np.matmul(np.linalg.inv(cameraMatrix), img_coor_mat_rs)
    cam_world_coor_mat = np.matmul(r_cam, cam_coor_mat_rs)
    # calculate intersect point
    scale_factor1 = -t_cam[2,0] / cam_world_coor_mat[2,:]
    scale_factor1 = np.reshape(scale_factor1, (1, scale_factor1.shape[0]))
    cam_mirror_intersect_mat = scale_factor1 * cam_world_coor_mat + np.tile(t_cam, (cam_world_coor_mat.shape[1]))
    # calculate reflectance vector in world coor system
    reflect_mat = cam_world_coor_mat - 2 * np.dot(np.array([0,0,1]), cam_world_coor_mat) * np.tile(np.array([[0],[0],[1]]), (1, cam_world_coor_mat.shape[1]))
    # calculate intersect point in display coor system
    mirror_intersect_mat_trans = np.dot(r_dis_inv, cam_mirror_intersect_mat) + np.tile(t_dis_inv, (cam_world_coor_mat.shape[1]))
    # calculate reflectance vector in display coor system
    reflect_mat_trans = np.dot(r_dis_inv, reflect_mat)
    # calculate display intersect point under display coordinate system
    scale_factor2 = -mirror_intersect_mat_trans[2,:] / reflect_mat_trans[2,:]
    display_intersect_mat = scale_factor2 * reflect_mat_trans + mirror_intersect_mat_trans
    # calculate display intersect point under world coordinate system
    display_intersect_mat_trans = np.dot(r_dis, display_intersect_mat) + np.tile(t_dis, (cam_world_coor_mat.shape[1]))

    # restore the matrix back to the same shape of undistort image
    display_intersect_mat = np.reshape(display_intersect_mat, (3, img_undistort_size[0], img_undistort_size[1]))
    display_intersect_mat_trans = np.reshape(display_intersect_mat_trans, (3, img_undistort_size[0], img_undistort_size[1]))
    cam_mirror_intersect_mat = np.reshape(cam_mirror_intersect_mat, (3, img_undistort_size[0], img_undistort_size[1]))

    #np.savetxt("displayintersect0.txt",display_intersect_mat[0,:,:])
    #np.savetxt("displayintersect1.txt",display_intersect_mat[1,:,:])

    ## this part is to display camera, mirror &projector within a unified world coordinate systems

    #reshape t vector
    t_cam = np.reshape(t_cam, (t_cam.shape[0],))
    t_dis = np.reshape(t_dis, (t_dis.shape[0],))
    t_dis_inv = np.reshape(t_dis_inv, (t_dis_inv.shape[0], ))

    #mirror corner index
    markerCornerIDs = np.array([np.argwhere(markerIDs == 0), np.argwhere(markerIDs == 2), np.argwhere(markerIDs == 4), np.argwhere(markerIDs == 6)])
    markerCornerIDs = markerCornerIDs[:,0,0]

    mirror_corner_value = np.array([markerCorners[markerCornerIDs[0]][0][0],
    markerCorners[markerCornerIDs[1]][0][3],
    markerCorners[markerCornerIDs[2]][0][2],
    markerCorners[markerCornerIDs[3]][0][1],
    markerCorners[markerCornerIDs[0]][0][0]], dtype=np.int32)
    #print("mirror corner index is", mirror_corner_value)
    mirror_corner_world = np.zeros((mirror_corner_value.shape[0],3))
    for i in range(mirror_corner_world.shape[0]):
        mirror_corner_world[i,:] = cam_mirror_intersect_mat[:, mirror_corner_value[i,1], mirror_corner_value[i,0]]
    #print("mirror corner point is", mirror_corner_world)

    #display corners in world coordinate system
    half_length_disp = 960 * displayScaleFactor
    half_height_disp = 540 * displayScaleFactor
    disp_uleft = np.array([-half_length_disp, -half_height_disp, 0])
    disp_lleft = np.array([-half_length_disp, half_height_disp, 0])
    disp_lright = np.array([half_length_disp, half_height_disp, 0])
    disp_uright = np.array([half_length_disp, -half_height_disp, 0])
    disp_uleft_world = np.reshape(np.matmul(r_dis, disp_uleft) + t_dis, (1, t_dis.shape[0]))
    disp_lleft_world = np.reshape(np.matmul(r_dis, disp_lleft) + t_dis, (1, t_dis.shape[0]))
    disp_lright_world = np.reshape(np.matmul(r_dis, disp_lright) + t_dis, (1,t_dis.shape[0]))
    disp_uright_world = np.reshape(np.matmul(r_dis, disp_uright) + t_dis, (1, t_dis.shape[0]))

    disp_corners_world = np.concatenate((disp_uleft_world, disp_lleft_world, disp_lright_world, disp_uright_world, disp_uleft_world), axis= 0)
    #print("disp corners in world coordinate is ", disp_corners_world)

    # calculate the (0,0) and z axis of camera and display (for verification)
    cam_z1 = np.matmul(r_cam, np.array([0,0,1])) + t_cam
    cam_zero = np.matmul(r_cam, np.array([0,0,0])) + t_cam
    display_z1 = np.matmul(r_dis,np.array([0,0,1])) + t_dis
    display_zero = np.matmul(r_dis,np.array([0,0,0])) + t_dis
    #cam_z = np.concatenate((np.reshape(cam_zero, (1, cam_zero.shape[0])), np.reshape(cam_z1, (1, cam_z1.shape[0]))), axis= 0)
    #display_z = np.concatenate((np.reshape(display_zero, (1,display_zero.shape[0])), np.reshape(display_z1, (1, display_z1.shape[0]))), axis= 0)
    cam_z = cam_z1 - cam_zero
    display_z = display_z1 - display_zero
    print('cam z axis in world coor is ',cam_z)
    print("cam zero point in world coor is ", cam_zero)
    print("display z axis in world coor is ", display_z)
    print("display zero in world coor is ", display_zero)
    # plot 
    fig = plt.figure(figsize=(15,10))
    ax = Axes3D(fig)
    ax.plot(mirror_corner_world[:,0], mirror_corner_world[:,1], mirror_corner_world[:,2], label= 'mirror')
    ax.plot(disp_corners_world[:,0], disp_corners_world[:,1], disp_corners_world[:,2], label= 'display')
    ax.quiver(cam_zero[0], cam_zero[1], cam_zero[2], cam_z[0], cam_z[1], cam_z[2], length= 0.4)
    ax.quiver(display_zero[0], display_zero[1], display_zero[2], display_z[0], display_z[1], display_z[2], length= 0.4)
    plt.title("Screen, Camera and Object in World Coordinate System")
    plt.legend()
    plt.show()
    imgPath = os.path.join(os.path.join(os.path.join(os.getcwd(), 'CalibrationImages'), 'Geometric'), 'geoCalibResults')
    fig.savefig(imgPath + '/Screen, Camera and Object in World Coordinate System' + '.png')

    return display_intersect_mat, cam_mirror_intersect_mat, rcamera_dis, tcamera_dis