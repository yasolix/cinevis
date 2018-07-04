
import glob
import os
from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import pandas as pd

pathmovies = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/ContinuousLIRIS-ACCEDE/continuous-movies/"
movframes = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/movframes/"
testsetframes = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/testsetframes/"

pathmovies = "/media/yt/Seagate Expansion Drive/2018laptop/ContinuousLIRIS-ACCEDE/continuous-movies/"
movframes = "/media/yt/Seagate Expansion Drive/2018laptop/cvpr2014/repro/mediaeval/data/dataset/movframes/"
#testsetframes = "/media/yt/Seagate Expansion Drive/2018laptop/cvpr2014/repro/mediaeval/data/dataset/testsetframes/"

lmfold = '/home/yt/Downloads/pnas_py3/landmarks/'
colfold= '/home/yt/Downloads/pnas_py3/colors/'

def findFaceInfo(filename, sep='\t'):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(filename)
    hframe, wframe = image.shape[0], image.shape[1]

    # Find all the faces in the image using a pre-trained convolutional neural network.
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
    numface = len(face_locations)
    if (sep=='\t'):
        face_landmarks_list = face_recognition.get_raw_face_landmarks(image, face_locations=face_locations)
    else:
        face_landmarks_list = face_recognition.face_landmarks(image,face_locations=face_locations)

    nummark = len(face_landmarks_list)

    return {'height' : hframe, 'width': wframe, 'numface':numface, 'locations': face_locations,'nummark': nummark, 'landmark':face_landmarks_list}


def facesinmovie(moviename,outfolder):

    files = sorted(glob.glob(movframes + moviename + '*.jpg'))

    cols = ["height", "width","numface", "locations", "nummark", "landmark"]

    facelist = []
    for idx, f in enumerate(files):
        print(idx,f)
        finfo = findFaceInfo(f)
        facelist.append(finfo)

    df = pd.DataFrame(facelist, columns = cols)
    facefile = outfolder + moviename + '-faces-info.txt'
    df.to_csv(facefile, sep='\t', header=True, index_label='id')
    return facelist


if __name__ == "__main__":
    #facesinmovie('Cloudland', '/home/yt/Downloads/pnas_py3/landmarks/')
    facesinmovie('Chatter', lmfold)

    f='/media/yt/Seagate Expansion Drive/2018laptop/cvpr2014/repro/mediaeval/data/dataset/movframes/Cloudland.mp4-00034.jpg'
    fl = findFaceInfo(f)
    cols = ["height", "width", "numface", "locations", "nummark", "landmark"]
    print(fl)
    df = pd.DataFrame(fl, columns=cols)

    print(df)

