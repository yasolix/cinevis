Thank you for downloading LIRIS-ACCEDE.

What LIRIS-ACCEDE is:
*********************

LIRIS-ACCEDE is the Annotated Creative Commons Emotional DatabasE.
LIRIS-ACCEDE is composed of three sections:
- discrete: original dataset composed of 9800 video clips extracted from 160 movies shared under Creative Commons licenses. It allows to make this database publicly available without copyright issues to contribute to the need for emotional databases and affective tagging. The 9800 video clips (each 8-12 seconds long) are sorted along the induced valence axis, from the video perceived the most negatively to the video perceived the most positively. The annotation was carried out by 1517 annotators from 89 different countries using crowdsourcing.
		Publication: Y. Baveye, E. Dellandrea, C. Chamaret, and L. Chen, “LIRIS-ACCEDE: A Video Database for Affective Content Analysis,” in IEEE Transactions on Affective Computing, 2015
		Files:
		- LIRIS-ACCEDE-data.zip
		- LIRIS-ACCEDE-annotations.zip
		- LIRIS-ACCEDE-movies.zip
		- LIRIS-ACCEDE-features
- continuous: 30 full movies continuously annotated along induced and valence axes. Physiological measurements are also available
		Publication: Y. Baveye, E. Dellandrea, C. Chamaret, and L. Chen, “Deep Learning vs. Kernel Methods: Performance for Emotion Prediction in Videos,” in ACII, 2015
		Publication: L. Ting, Y. Baveye, C. Chamaret, E. Dellandrea, and L. Chen, “Continuous Arousal Self-assessments Validation Using Real-time Physiological Responses,” in ASM (ACM MM workshop), 2015
		Files:
		- LIRIS-ACCEDE-continuous-movies.zip
		- LIRIS-ACCEDE-continuous-annotations.zip
		- LIRIS-ACCEDE-continuous-Gtrace.zip
		- LIRIS-ACCEDE-continuous-physiological.zip
- MediaEval dataset: used for the MediaEval 2015 affective impact of movies task. Violence ratings and affective classes are available for the 9800 excerpts of the original discrete LIRIS-ACCEDE, and for 1100 additional video excerpts.
		Publication: M. Sjöberg, Y. Baveye, H. Wang, V. L. Quang, B. Ionescu, E. Dellandréa, ... & L. Chen, “The mediaeval 2015 affective impact of movies task,” in MediaEval 2015 Workshop, 2015
		Files:
		- MEDIAEVAL-data.zip
		- MEDIAEVAL-annotations.zip
		- MEDIAEVAL-movies.zip

How to use LIRIS-ACCEDE:
************************

LIRIS-ACCEDE-data.zip
_ /data
    The 9800 mp4 video excerpts are saved in the data folder.
_ ACCEDEdescription.xml
    This file contains the description of each video excerpt.
    For example the description of the first video excerpt is:
      <media>
        <id>0</id>
        <name>ACCEDE00000.mp4</name>
        <license>20 Mississippi shared under Creative Commons Attribution-NonCommercial 3.0 Unported licence at http://vimeo.com/20043857</license>
        <movie>20_Mississippi.mp4</movie>
        <start>833</start>
        <end>1094</end>
      </media>
    _ <id> is the identifier of the excerpt
    _ <name> is the file name of the excerpt saved in /data
    _ <license> is the license and the download link where the original movie can be downloaded
    _ <movie> is the file name of the original movie (ask us if you want them)
    _ <start> is the number of the first frame of the excerpt in the original movie
    _ <end> is the number of the last frame of the excerpt in the original movie
      NB: 1000 video excerpts have been manually segmented and their locations in the original movie have been retrieved using an algorithm. Thus, because they have been manually segmented, different excerpts may be interlaced (the end frame of an excerpt bigger than the start frame of the next excerpt). The movies composed of excerpts manually extracted are: the 52 movies "52_Films_52_Weeks", "Big_Buck_Bunny", "Damaged_Kung_Fu", "Elephant_s_Dream", "Je_Suis_Ce_Que_Je_Vois", "Sintel" and "The_Cosmonaut_Trailer".

LIRIS-ACCEDE-annotations.zip
_ /annotations/ACCEDEranking.txt
	This file is composed of 8 columns:
	_ id: identifier of the excerpt
	_ name: file name of the excerpt saved in /data
	_ valenceRank: rank of the excerpt in the database along the valence axis
	_ arousalRank: rank of the excerpt in the database along the arousal axis
	_ valenceValue: valence score computed by the regression
	_ arousalValue: arousal score computed by the regression
	_ valenceVariance: variance computed by the regression for valence
	_ arousalVariance: variance computed by the regression for arousal
    The smallest rank (i.e. 0) corresponds to the excerpt with the lowest valence/arousal.
    Whereas the highest rank (i.e. 9799) corresponds to the excerpt with the highest valence/arousal.
	Valence and arousal values can range from 1 to 5.
	Computational details for scores and variance values can be found in "Yoann Baveye, Emmanuel Dellandréa, Christel Chamaret and Liming Chen, ‘‘From Crowdsourced Rankings To Affective Ratings,’’ MAC 2014, 2014."
_ /annotations/ACCEDEsets.txt
    For each video clip identified by its id and its name, this files tells if the video excerpts has been used in:
	_ the validation set (value 2): 2450 excerpts extracted from 40 movies
	_ the learning set (value 1): 2450 excerpts extracted from 40 movies
	_ or test set (value 0): 4900 excerpts extracted from 80 movies
	In case the validation set is not used, it is included in the learning set. Thus, the learning set (called FULL learning set) is in this case composed of excerpts with value 1 or 2 (4900 excerpts extracted from 80 movies).

LIRIS-ACCEDE-movies.zip
_ /movies
    The 175 original video mp4 files used to segment the 9800 excerpts.
_ ACCEDEmovies.xml
	This file contains the description of each original video file.
    For example the description of the first video file is:
      <media>
		<movie>20_Mississippi.mp4</movie>
		<title>20 Mississippi</title>
		<license>20 Mississippi shared under Creative Commons Attribution-NonCommercial 3.0 Unported license at http://vimeo.com/20043857</license>
		<credits>Michael Brettler</credits>
		<genre>Drama</genre>
		<language>English</language>
		<length>00:58:38</length>
		<excerpts>145</excerpts>
	  </media>
    _ <movie> is the file name of the original movie (ask us if you want them)
      NB: Video excerpts are extracted from 160 movies but there are 175 different movie file names since several movies are splitted in several files (i.e. the movie "The Dabbler" is splitted into The_Dabbler_1_6.mp4, The_Dabbler_2_6.mp4, The_Dabbler_3_6.mp4, The_Dabbler_4_6.mp4, The_Dabbler_5_6.mp4 and The_Dabbler_6_6.mp4)
    _ <title> is the original title of the movie
    _ <license> is the license and the download link where the original movie can be downloaded
	_ <credits> references the original creator(s)
	_ <genre> is the genre of the movie
	_ <language> is the language of the movie
	_ <length> is the length of the mp4 file
	_ <excerpts> is the number of excerpts extracted from the mp4 file

LIRIS-ACCEDE-features.zip
_ /features/ACCEDEfeaturesValence_ACII2013.txt
	This file includes all the features (see description below) used in [Baveye 2013] (see references below)
_ /features/ACCEDEfeaturesArousal_TAC2015.txt
	This file includes all the features used to learn the model to predict arousal values in [Baveye 2015]
_ /features/ACCEDEfeaturesValence_TAC2015.txt
	This file includes all the features used to learn the model to predict valence values in [Baveye 2015]
_ General Description:
	In all these three files, each column represents the value of a feature extracted from the excerpt indicated in the first column.
	Here is a description of the features:
	_ alpha: orientation of the harmonious template described in [Baveye 2013] or [Baveye 2015]
	_ asymmetry: audio asymmetry described in [Baveye 2013]
	_ asymmetry_env: audio asymmetry envelop described in [Baveye 2015]
	_ centroid: audio frequency centroid described in [Baveye 2013]
	_ colorfulness: colorfulness described in [Baveye 2013] or [Baveye 2015]
	_ colorRawEnergy: color energy described in [Baveye 2015]
	_ colorStrength: color contrast described in [Baveye 2015]
	_ compositionalBalance: compositional balance described in [Baveye 2015]
	_ cutLength: length of scene cuts described in [Baveye 2015]
	_ depthOfField: depth of field described in [Baveye 2015]
	_ energy: audio energy described in [Baveye 2013]
	_ entropyComplexity: entropy complexity described in [Baveye 2013] or [Baveye 2015]
	_ flatness: audio flatness described in [Baveye 2013] or [Baveye 2015]
	_ flatness_env: audio flatness envelop described in [Baveye 2015]
	_ globalActivity: global activity described in [Baveye 2015]
	_ hueCount: hue count described in [Baveye 2015]
	_ lightning: lightning feature described in [Baveye 2015]
	_ maxSaliencyCount: number of max salient pixels in [Baveye 2013] or [Baveye 2015]
	_ medianLightness: median lightness described in [Baveye 2015]
	_ minEnergy: energy corresponding to the most harmonious template described in [Baveye 2015]
	_ nbFades: number of fades per frame described in [Baveye 2015]
	_ nbSceneCuts: number of scene cuts per frame described in [Baveye 2015]
	_ nbWhiteFrames: normalized number of white frames described in [Baveye 2015]
	_ saliencyDisparity: disparity of salient pixels described in [Baveye 2013] or [Baveye 2015]
	_ spatialEdgeDistributionArea: spatial edge distribution area described in [Baveye 2013] or [Baveye 2015]
	_ stdDevLocalMax: standard deviation of local maxima described in [Baveye 2013]
	_ roll_off: spectral roll-off described in [Baveye 2013]
	_ wtf_max2stdratio (12 features): standard deviation of the wavelet coefficients of the audio signal described in [Baveye 2015]
	_ zcr: zero-rossing rate described in [Baveye 2013] or [Baveye 2015]

LIRIS-ACCEDE-continuous-movies.zip
_ /continuous-movies
	The 30 movies from which continuous arousal and valence ratings have been collected.
	These 30 full movies are part of the discrete LIRIS-ACCEDE dataset. Thus, for these 30 movies continuous ratings as well as discrete ranks and discrete ratings collected in [Baveye 2013] are available.
	
LIRIS-ACCEDE-continuous-annotations.zip
_ /continuous-annotations
	Two files per movie: 'movieName_Arousal.txt' and 'movieName_Valence.txt' including for each second of a movie the post-processed average of the ratings of the annotators as detailed in [Baveye 2015-2].
	Original annotation files per annotator are available in the folder /continuous-annotations/raw
_ /scripts
	All the python scripts included in this folder have been developed with Python 3.3 and may not work with Python 2.7
	_ ACCEDEcomparisonDiscreteContinuousAnalysis.py generates figure to superimpose discrete and continuous ratings
	_ The 1 second long segments used in [Baveye 2015-2] can be generated using the script: "ACCEDEsplitMoviesForContinuousAnalysis.py"
	  Please note that for this script, FFMPEG should be installed and in your working environment in order to be recognized by Python.
	
LIRIS-ACCEDE-continuous-Gtrace.zip
_ /continuous-Gtrace/Readme.pdf
	This file explains how to use the modified Gtrace software and the changes that have been made
_ /continuous-Gtrace/Trace2011v20140611
	Contains the sources of the modified software originally developed by cowie et al and originally available here: https://sites.google.com/site/roddycowie/work-resources

LIRIS-ACCEDE-continuous-physiological.zip
_ /continuous-physiological
	Contains one csv file for each of the movies used in the continuous experiment
	Each CSV file contains 3 columns separated by semicolons.
	The first column is the second of the movie associated to the post-processed GSR measure (second column) and arousal value (third column).
	More information about the post-processing of the GSR measurements can be found in [Ting 2015].
	Please not that raw measurements are not available because they are saved using a proprietary format.

MEDIAEVAL-data.zip
_ /data
    The 1100 mp4 video excerpts used for the test set (in addition to the elements in the test set defined in ACCEDEsets.txt) for the MediaEval 2015 affective impact of movies task are saved in the data folder.
_ MEDIAEVALdescription.xml
	Similar to ACCEDEdescription.xml, but for the movies used to build the MediaEval test set.
	
MEDIAEVAL-annotations.zip
_ /annotations/MEDIAEVALaffect.txt
	This file is composed of 4 columns:
	_ id: identifier of the excerpt
	_ name: file name of the excerpt saved in /data
	_ valenceClass: the valence class of the excerpt (-1 for negative/0 for neutral/1 for positive)
	_ arousalClass: the arousal class of the excerpt (-1 for calm/0 for neutral/1 for arousing)
	These values are indicated for each of the 1100 video clips in the MediaEval test set.
	More information about the collection of the annotations can be found in [Sjöberg 2015].
_ /annotations/MEDIAEVALviolence.txt
	For each of the 1100 video clips in the MediaEval test set, identified by its id and its name, this files tells if the video excerpts is:
	_ violent (value 1)
	_ non violent (value 0)
	More information about the collection of the annotations can be found in [Sjöberg 2015].
_ /annotations/MEDIAEVALsets.txt
	Similar to /annotations/ACCEDEsets.txt
_ /annotations/ACCEDEaffect.txt
	Similar to /annotations/MEDIAEVALaffect.txt but for the 9800 videos of the discrete LIRIS-ACCEDE.
	Affect classes for this file have been generated from the estimated affect score from /annotations/ACCEDEranking.txt.
	More information about the conversion from estimated affect scores to affect classes can be found in [Sjöberg 2015].
_ /annotations/ACCEDEviolence.txt
	For each of the 9800 video clips in the discrete LIRIS-ACCEDE, identified by its id and its name, this files tells if the video excerpts is:
	_ violent (value 1)
	_ non violent (value 0)
	More information about the collection of the annotations can be found in [Sjöberg 2015].

MEDIAEVAL-movies.zip
_ /movies
    The 42 original video mp4 files used to segment the 1100 excerpts.
_ MEDIAEVALmovies.xml
	Similar to ACCEDEmovies.xml

Changelog:
**********

15/10/2015
	Add continuous ratings
	Add physiological (GSR) measurements
	Add MediaEval dataset with violence annotations
	Add modified Gtrace software to collect continuous ratings
	Add original movie files for the whole LIRIS-ACCEDE dataset (discrete, continuous and MediaEval test set)

25/11/2014
	Add the ranks converted into ratings
	Add the features used in the ACII paper

08/08/2014
	Merge of the 4 movies that were split in several files (i.e. the movie "The Dabbler" has been merged into the file The_Dabbler.mp4. It was splitted into The_Dabbler_1_6.mp4, The_Dabbler_2_6.mp4, The_Dabbler_3_6.mp4, The_Dabbler_4_6.mp4, The_Dabbler_5_6.mp4 and The_Dabbler_6_6.mp4)
	Add estimated rating values to ACCEDEranking.txt
	Add missing frame numbers to ACCEDEdescription.xml
	Add credits information to ACCEDEmovies.xml
	Add validation set to ACCEDEsets.txt (and switch 52_Films_52_Weeks_I_m_Dead and 52_Films_52_Weeks_Irving_J__Koppermelt between test and learning sets)

04/11/2013
	Initial release

References:
***********

[Baveye 2013] Y. Baveye, J.-N. Bettinelli, E. Dellandrea, L. Chen, and C. Chamaret, “A large video data base for computational models of induced emotion,” in Affective Computing and Intelligent Interaction (ACII), 2013 Humaine Association Conference on, 2013, pp. 13–18.
[Baveye 2015] Y. Baveye, E. Dellandrea, C. Chamaret, and L. Chen, “LIRIS-ACCEDE: A Video Database for Affective Content Analysis,” in IEEE Transactions on Affective Computing, 2015
[Baveye 2015-2] Y. Baveye, E. Dellandrea, C. Chamaret, and L. Chen, “Deep Learning vs. Kernel Methods: Performance for Emotion Prediction in Videos,” in ACII, 2015
[Ting 2015] Y. Baveye, C. Chamaret, E. Dellandrea, and L. Chen, “Continuous Arousal Self-assessments Validation Using Real-time Physiological Responses,” in ASM (ACM MM workshop), 2015
[Sjöberg 2015] M. Sjöberg, Y. Baveye, H. Wang, V. L. Quang, B. Ionescu, E. Dellandréa, ... & L. Chen, “The mediaeval 2015 affective impact of movies task,” in MediaEval 2015 Workshop, 2015