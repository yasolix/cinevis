### To extract images for each second from the movie
ffmpeg -loglevel error -i Islands.mp4 -r 1 -f image2 movframes/Islands.mp4-%05d.jpg

### To capture shot boundary/scene change  image
ffmpeg -i Islands.mp4 -filter:v "select='gt(scene,0.1)',showinfo" -vsync 0 movframes/IslandsShots/shot-%05d.jpg

### To get the time stamps of the scene changes
ffmpeg -i Islands.mp4 -filter:v "select='gt(scene,0.4)',showinfo" -vsync 0 movframes/IslandsShots2/shot-%05d.jpg 2>ffout
grep showinfo ffout | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > timestamps

####  To grab both floating point and integer timestamps use the following regex
grep showinfo ffout | grep pts_time:[0-9.]* -o | grep -E '[0-9]+(?:\.[0-9]*)?' -o > timestamps2

### Create a mosaic of the first scenes
ffmpeg -i video.avi -vf select='gt(scene\,0.4)',scale=160:120,tile -frames:v 1 preview.png

### To visualize the waveforms of the video
ffplay -i Big_Buck_Bunny.mp4 -vf "split[a][b];[a]format=gray,waveform,split[c][d];[b]pad=iw:ih+256[padded];[c]geq=g=1:b=1[red];[d]geq=r=1:b=1,crop=in_w:220:0:16[mid];[red][mid]overlay=0:16[wave];[padded][wave]overlay=0:H-h"

### FFMPEG scene detection documentation
http://www.ffmpeg.org/ffmpeg-filters.html#select_002c-aselect

resources
stackoverflow
http://trac.ffmpeg.org/wiki/FancyFilteringExamples
http://www.bogotobogo.com/FFMpeg/ffmpeg_thumbnails_select_scene_iframe.php
