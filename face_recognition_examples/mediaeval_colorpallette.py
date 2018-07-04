
##https://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/

import pandas as pd
import glob
from collections import namedtuple
from math import sqrt
import random
try:
    import Image
except ImportError:
    from PIL import Image



Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

pathmovies = "/media/yt/Seagate Expansion Drive/2018laptop/ContinuousLIRIS-ACCEDE/continuous-movies/"
movframes = "/media/yt/Seagate Expansion Drive/2018laptop/cvpr2014/repro/mediaeval/data/dataset/movframes/"

lmfold = '/home/yt/Downloads/pnas_py3/landmarks/'
colfold= '/home/yt/Downloads/pnas_py3/colors/'


def get_points(img):
    points = []
    w, h = img.size
    for count, color in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(filename, n=3):
    img = Image.open(filename)
    img.thumbnail((200, 200))
    w, h = img.size

    points = get_points(img)
    if (len(points)<=10*n):
        return ['#000000' for i in range(n)]
    clusters = kmeans(points, n, 1)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([
        (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
    ]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0.0001
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters



def colorsinmovie(moviename):

    files = sorted(glob.glob(movframes + moviename + '*.jpg'))

    colorlist = []
    for idx, f in enumerate(files):
        print(idx,f)
        c = colorz(f,n=5)
        colinfo = list()
        for x in c:
            colinfo.append(x)
        colorlist.append(colinfo)

    colorfile = colfold + moviename + '-color-info.txt'
    df = pd.DataFrame(colorlist)
    df.to_csv(colorfile, sep='\t')

    return colorlist


if __name__ == "__main__":

    selected=['After_The_Rain',  'Barely_legal_stories', 'Damaged_Kung_Fu', 'Payload', 'Sintel', 'Tears_of_Steel', 'The_secret_number']

    for movie in selected:
        df = colorsinmovie(movie)

    #df = colorsinmovie('Chatter')
    #df = colorsinmovie('Cloudland')
    f='/media/yt/Seagate Expansion Drive/2018laptop/cvpr2014/repro/mediaeval/data/dataset/movframes/Cloudland.mp4-00034.jpg'
    a = colorz(f)
    for c in a:
        print(c,type(c))
