import sys, glob, os
import argparse
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt


#sqWidth = 11 # number of squares width
#sqHeight = 8 # number of squares height
#checkerSquareSize = 0.070 #0.024 # size of the checker square in (meter)
#markeSquareSize = 0.052 #0.018 # size of the marker square in (meter)

def saveCameraParams(filename,imageSize,cameraMatrix,distCoeffs,totalAvgErr):

	print(cameraMatrix)

	calibration = {'camera_matrix': cameraMatrix.tolist(),'distortion_coefficients': distCoeffs.tolist()}

	calibrationData = dict(
			 image_width = imageSize[0],
			 image_height = imageSize[1],
			 camera_matrix = dict(
				 rows = cameraMatrix.shape[0],
				 cols = cameraMatrix.shape[1],
				 dt = 'd',
				 data = cameraMatrix.tolist(),
				 ),
			 distortion_coefficients = dict(
					 rows = disCoeffs.shape[0],
					 cols = disCoeffs.shape[1],
					 dt = 'd',
					 data = disCoeffs.tolist(),
					 ),
			 avg_reprojection_error = totalAvgErr,)

	with open(filename,'w') as outfile:
			 yaml.dump(calibrationData,outfile)

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

def calibration(imgFolder, imgDistortFolder, calibFname, sqWidth, sqHeight, checkerSquareSize, markeSquareSize):

	imgFileList = readFileList(imgFolder)

	allCorners = [] #all Charuco Corners
	allIds = [] #all Charuco Ids
	decimator = 0
	#cameraMatrix = np.array([])
	#disCoeffs = np.array([])
	#viewErrors = np.array([])
	dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250) # maker sure you use the right dictionary
	board = cv2.aruco.CharucoBoard_create(sqWidth,sqHeight,checkerSquareSize,markeSquareSize,dictionary)

	for i in imgFileList:

			# print("reading %s" % i)
			img = cv2.imread(i)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cornerRefine = cv2.aruco.CORNER_REFINE_CONTOUR

			[markerCorners,markerIds,rejectedImgPoints] = cv2.aruco.detectMarkers(gray,dictionary)#, parameters=cv2.aruco.CORNER_REFINE_CONTOUR)


			if len(markerCorners)>0:
					[ret,charucoCorners,charucoIds] = cv2.aruco.interpolateCornersCharuco(markerCorners,markerIds,gray,board)
					if charucoCorners is not None and charucoIds is not None and len(charucoCorners)>3:# and decimator%3==0:
							allCorners.append(charucoCorners)
							allIds.append(charucoIds)

					cv2.aruco.drawDetectedMarkers(img,markerCorners,markerIds, [0, 255, 0])
					cv2.aruco.drawDetectedCornersCharuco(img,charucoCorners,charucoIds, [0, 0, 255])

					#for corner in allCorners:
					#    cv2.circle(img,(corner[0][0], corner[0][0]),50,(255,255,255))
			width = 960
			height = int(img.shape[0]*960/img.shape[1])
			smallimg = cv2.resize(img,(width,height))
			#plt.figure()
			
			#plt.imshow(smallimg)
			imgSave = os.path.join(os.path.join(imgFolder, 'calibResults'), str(decimator) + '.png') 
			cv2.imwrite(imgSave, smallimg)
			#plt.savefig('tessstttyyy.png', dpi=100)
			#plt.show()
			
			decimator+=1
	print("NumImg:", len(allCorners))
	imsize = img.shape
	print(imsize)
	#try Calibration
	try:
		#, aa, bb, viewErrors
			[ret, cameraMatrix, disCoeffs, rvecs, tvecs, _, _, perViewErrors] = cv2.aruco.calibrateCameraCharucoExtended(allCorners,allIds,board,(imsize[0], imsize[1]),
				None,None,flags=(cv2.CALIB_RATIONAL_MODEL), criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
			print("Reprojection Error:" ,ret)
			print("Camera Intrinsic Matrix:" ,cameraMatrix)
			#print("per View Errors:" ,perViewErrors)
			print("Distortion Coeffs:" ,disCoeffs)
			#print("Rvecs:", rvecs)
			#np.savez(calibFname,ret=ret,mtx=cameraMatrix,dist=disCoeffs,rvecs=rvecs,tvecs=tvecs)
			#saveCameraParams(args.file,imsize,cameraMatrix,disCoeffs,ret)
			#try to undistort the images
			#please change the folder of images to be undistorted
			
			#imgDistortFolder = '/Users/li/Documents/Deflectometry/rcalibration0418/large1117_3'
			imgDistortFilelist = readFileList(imgDistortFolder)
			img_num = 0
			for j in imgDistortFilelist:

				imgDistort = cv2.imread(j)
				imgDistortgray = cv2.cvtColor(imgDistort, cv2.COLOR_BGR2GRAY)
			#print(imgDistort.shape)
			#h, w = imgDistort.shape[:,2]
				h = imgDistort.shape[0]
				w = imgDistort.shape[1]
				newcameramtx,roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, disCoeffs, (w, h), 1, (w, h))
				#print('image cropped', j)
			#undistort
				#dst = cv2.undistort(imgDistort, cameraMatrix, disCoeffs, None, newcameramtx)
				dst = cv2.undistort(imgDistort, cameraMatrix, disCoeffs, None)
				print('image undistorted', j)
				x,y,w,h = roi
				#dst = dst[y:y + h, x:x + w]
				imgDistortFilename = os.path.join(imgDistortFolder,'undistort', str(img_num) + '.png')
				cv2.imwrite(imgDistortFilename, dst)
				#print('image saved', j)
				img_num = img_num + 1
			

	except ValueError as e:
			print(e)
	except NameError as e:
			print(e)
	except AttributeError as e:
			print(e)
	except:
			print("calibrateCameraCharuco fail:" , sys.exc_info()[0])

	print("Intrinsic calibration complete!")
	return ret, cameraMatrix, disCoeffs

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("-f", "--file", help="ouput calibration filename",default="calibration.npz")
	#parser.add_argument("-s", "--size", help="size of squares in meters",type=float, default="0.035")
	parser.add_argument('-p', "--path", type=str, help='path of images for calibration')
	args = parser.parse_args()
	calibration(os.path.normpath(args.path), os.path.join(args.path, args.file))



