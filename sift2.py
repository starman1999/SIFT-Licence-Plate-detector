import cv2.cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

MIN_MATCH_COUNT = 10
cpt=0


img2 = cv2.imread('Project\queries\matricule1.png')
print(img2)

model = []
train = []

for x in glob.iglob('Projec\License Plates\*'):
    image2 = cv2.imread(x)
    train.append(image2)

for f in glob.iglob('queries\*'):
    image = cv2.imread(f)
    model.append(image)

for image_train in train:



   for image_to_compare in model:

   # Initiate SIFT detector
    sift = cv2.SIFT_create()

   # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_to_compare, None)
    kp2, des2 = sift.detectAndCompute(image_train, None)



  # matching algorythm
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2) #(tuple with matches)

     # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
     if m.distance < 0.75*n.distance:
         good.append(m)

    if len(good) > MIN_MATCH_COUNT:


     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
     matchesMask = mask.ravel().tolist()
     print('the same')

     draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2
     )

     img3 = cv2.drawMatches(image_to_compare, kp1, image_train, kp2, good, None, **draw_params)

     cv2.imshow(f'car Number{cpt + 1}:', image_to_compare)
     cv2.waitKey()
     cv2.destroyAllWindows()
     plt.imshow(img3, cmap='gray', interpolation='bicubic')
     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
     plt.show()

    else:
     print("not the same license plate ! -- Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )

   cpt=cpt+1
   print('next LP',cpt)




