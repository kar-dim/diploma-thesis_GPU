REM download sample video, example https://media.xiph.org/video/derf/y4m/FourPeople_1280x720_60.y4m
REM convert to 30 fps (optional)
ffmpeg -i FourPeople_1280x720_60.y4m -filter:v fps=fps=30 FourPeople_1280x720_30.y4m
REM remove metadata and headers ([FRAME], [/FRAME} etc), to produce a raw-only data file:
ffmpeg -i FourPeople_1280x720_30.y4m -map_metadata -1 FourPeople_1280x720_30.yuv
REM compress the raw video of the previous step with libx265 and crf 28, and use AVI container (for OpenCV compatibility)
ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 30 -pix_fmt yuv420p -i FourPeople_1280x720_30.yuv -c:v libx265 -crf 28 FourPeople_1280x720_30_compressed_incomplete.avi
REM add the missing headers to the compressed file
ffmpeg -i FourPeople_1280x720_30_compressed_incomplete.avi -c copy -vtag hev1 -strict -2 FourPeople_1280x720_30_compressed.avi
REM delete files not needed:
del FourPeople_1280x720_30_compressed_incomplete.avi
REM We can use ffprobe to get number of frames and fram-type (I,P etc) info
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 FourPeople_1280x720_30_compressed.avi
pause