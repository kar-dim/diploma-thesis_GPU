REM compress the raw video of the previous step with libx265 and crf 28, and use AVI container (for OpenCV compatibility)
ffmpeg -f rawvideo -vcodec rawvideo -s 1280x720 -r 30 -pix_fmt yuv420p -i FourPeople_1280x720_30_watermarked_first_frame_only.yuv -c:v libx265 -crf 28 FourPeople_1280x720_30_compressed_watermarked_first_frame_only_incomplete.avi
REM add the missing headers to the compressed file
ffmpeg -i FourPeople_1280x720_30_compressed_watermarked_first_frame_only_incomplete.avi -c copy -vtag hev1 -strict -2 FourPeople_1280x720_30_compressed_watermarked_first_frame_only.avi
del FourPeople_1280x720_30_compressed_watermarked_first_frame_only_incomplete.avi
pause