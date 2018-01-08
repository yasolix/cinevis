import glob
import os
from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import pandas

pathmovies = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/ContinuousLIRIS-ACCEDE/continuous-movies/"
movframes = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/movframes/"
testsetframes = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/testsetframes/"

def findFace(filename):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(filename)
    hframe, wframe = image.shape[0], image.shape[1]

    top, right, bottom, left = 0, 0, 0, 0

    # Find all the faces in the image using a pre-trained convolutional neural network.
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location

    #returning the face information , (the last one if multiple face exist)
    return [len(face_locations), top, left, bottom, right, hframe, wframe]



def enlargebox(top,left,bottom,right, scale=0.25,maxh=0,maxw=0):
    # to enlarge the face detection box
    if top - top*scale < 0:
        top = 0
    else:
        top = top - np.floor(top*scale)
    if left - np.floor(left*scale) < 0:
        left = 0
    else:
        left = left - left*scale
    if bottom + bottom*scale > maxh:
        bottom = maxh
    else:
        bottom = bottom + np.floor(bottom*scale)
    if right + right*scale > maxw:
        right = maxw
    else:
        right = right + np.floor(right*scale)

    return int(top), int(left), int(bottom), int(right)


def makeupFace(filename,enlarge=False):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(filename)

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    hframe, wframe = image.shape[0], image.shape[1]
    #print("Frame width, height are {}, {}".format(hframe,wframe))
    #print("I found {} face(s) in this photograph.".format(len(face_locations)))

    aframe = hframe*wframe

    face_ratio=0

    hface, wface = 0, 0
    chface, cwface =0,0

    top, right, bottom, left = 0,0,0,0

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        # You can access the actual face itself like this:
        #face_image = image[top:bottom, left:right]
        #pil_image = Image.fromarray(face_image)
        #pil_image.show()

        if (enlarge):
            btop, bleft, bbottom, bright = enlargebox(top, left, bottom, right, 0.25, hframe, wframe)
            print(btop, bleft, bbottom, bright)
            face_image = image[btop:bbottom, bleft:bright]
            cv2.rectangle(image, (bleft, btop), (bright, bbottom), (0, 0, 255), 2)
        else:
            face_image = image[top:bottom, left:right]

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        pil_image = Image.fromarray(image)

        hface, wface = face_image.shape[0], face_image.shape[1]

        face_ratio = hface*wface*1.0/aframe

        print('{} {} Ratio of face = {}'.format(hface,wface,hface*wface*1.0/aframe))

        pil_image.show()

        # Find all the faces in the image using the default HOG-based model.
        # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(face_image)

        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

        for face_landmarks in face_landmarks_list:

             # Print the location of each facial feature in this image
             facial_features = [
                 'chin',
                 'left_eyebrow',
                 'right_eyebrow',
                 'nose_bridge',
                 'nose_tip',
                 'left_eye',
                 'right_eye',
                 'top_lip',
                 'bottom_lip'
             ]

             for facial_feature in facial_features:
                 print("The {} in this face has the following points: {}".format(facial_feature,
                                                                                 face_landmarks[facial_feature]))

             # Let's trace out each facial feature in the image with a line!
             pil_image = Image.fromarray(face_image)
             d = ImageDraw.Draw(pil_image)

             for facial_feature in facial_features:
                 d.line(face_landmarks[facial_feature], width=5)

             pil_image.show()

        chface = top +(bottom-top)/2
        cwface = left + (right-left)/2

    return [len(face_locations), chface/hframe, cwface/wframe, top, left, bottom, right, hframe, wframe ]

def find_faces_in_picture_cnn(filename):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(filename)

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()


def find_face_in_picture(filename):
    #image = face_recognition.load_image_file("biden.jpg")
    image = face_recognition.load_image_file(filename)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()


### write to file with numpy or Pandas
def facesinbatch(moviename):
    files = sorted(glob.glob(movframes + moviename + '*.jpg'))
    facefile = '/home/yt/cinevis/data/dlibfaces/'+moviename+'-faces.txt'
    ff = open(facefile,'w')
    for idx,f in enumerate(files):
        print(idx,f)
        fl = makeupFace(f)
        sfl = [str(l) + ' ' for l in fl]
        ff.write(str(idx+1)+' ')
        [ff.write(s) for s in sfl]
        ff.write('\n')
    ff.close()


def facesinmovie(moviename,istest=False,outfolder='/home/yt/cinevis/data/dlibfaces/'):
    if (istest):
        files = sorted(glob.glob(testsetframes + moviename +'/*.jpg'))
    else:
        files = sorted(glob.glob(movframes + moviename + '*.jpg'))

    flist = []
    for idx,f in enumerate(files):
        #print(idx,f)
        fl = findFace(f)
        print(fl)
        flist.append(fl)

    cols = ["noface","top", "left", "bottom", "right", "hframe", "wframe"]
    pd = pandas.DataFrame(flist,columns=cols)

    # Now you have a csv with columns and index:
    facefile = outfolder+moviename+'-faces.txt'
    pd.to_csv(facefile, sep=' ', header=True, index_label='id')


def find_facial_features_in_picture(filename = None, model=None):

    if (filename != None ):
        # Load the jpg file into a numpy array
        image = face_recognition.load_image_file(filename)

    # Find all facial features in all the faces in the image
    if (model != None):
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    else:
        face_locations = None

    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    pil_image = Image.fromarray(image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!

        d = ImageDraw.Draw(pil_image)

        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], width=5)

    pil_image.show()


### testing functions

#filename = movframes + 'Sintel' + '.mp4-00639.jpg' #'.mp4-00127.jpg'
filename = movframes + 'Attitude_Matters' + '.mp4-00111.jpg'
filename = movframes + 'Decay' + '.mp4-02534.jpg' #'.mp4-00625.jpg' # '.mp4-00274.jpg'
#filename = '/home/yt/Downloads/phd/progress2017/metin/images/upintheair2.png'

# HOG models
#find_face_in_picture(filename)
#find_facial_features_in_picture(filename)

#CNN models
find_faces_in_picture_cnn(filename)
find_facial_features_in_picture(filename, model='cnn')

#makeupFace()
#facesinbatch('Islands')
#facesinbatch('Sintel')
#facesinbatch('Big_Buck_Bunny')
#facesinbatch('Chatter')
#facesinbatch('Attitude_Matters')

#facesinmovie('After_The_Rain')

'''
movies = sorted([os.path.basename(x) for x in glob.glob(pathmovies+'*')])
print(movies)
for m in movies:
    print(m)
    facesinmovie(m)
'''
'''
testmovies = sorted([os.path.basename(x) for x in glob.glob(testsetframes+'*')])
print(testmovies)
for m in testmovies:
    print(m)
    facesinmovie(m, istest=True,outfolder='/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/dlibfaces/')
'''
