Thank you for downloading LIRIS-ACCEDE dataset.
This file contains visual features for the LIRIS-ACCEDE continuous part that is used for the MEDIAEVAL 2017 Emotional Impact of Movies task.
For each of the 30 movies, one frame per second is extracted using the following ffmpeg command: "ffmpeg -loglevel error -i movie.mp4 -r 1 -f image2 frame-%05d.jpg".

For each of these images, several general purpose visual features are provided. They have been extracted using LIRE library (http://www.lire-project.net/), except CNN features (vgg16 fc6 layer output) that have been extracted using Matlab Neural Networks toolbox.

- Auto Color Correlogram (acc)

- Color and Edge Directivity Descriptor (cedd)

- Color Layout (cl)

- Edge Histogram (eh)

- Fuzzy Color and Texture Histogram (fcth)

- Gabor (gabor)

- Joint descriptor joining CEDD and FCTH in one histogram (jcd)

- Scalable Color (sc)

- Tamura (tamura)

- Local Binary Patterns (lbp) 

- VGG16 fc6 layer output (fc6)

Features are stored in text files, one file by image and by feature type. For example, "After_The_Rain-00001_acc.txt" contains Auto Color Correlogram features (acc) for the 1st frame of After_The_Rain movie. Each feature file contains one line where feature values are separated by commas.

LIRE demo
Set of GlobalFeatures: 
ACCID
JCD
AutoColorCorrelogram
ColorLayout
PHOG
CEDD
SimpleColorHistogram
Gabor
OpponentHistogram
JointHistogram
COMO
LuminanceLayout
Tamura
FCTH
ScalableColor
EdgeHistogram
