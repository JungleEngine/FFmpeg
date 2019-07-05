#!/bin/sh
ffmpeg -framerate $2 -pattern_type glob -i $1'/*.jpg' -vcodec libx264 -pix_fmt yuvj420p generated.mp4 -y

