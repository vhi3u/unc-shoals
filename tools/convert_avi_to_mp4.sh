#!/bin/bash
# convert_avi_to_mp4.sh
#
# Convert AVI animations (from the vishoals.m fallback) to MP4 — no MATLAB needed.
#
# Usage:
#   ./convert_avi_to_mp4.sh                       # convert every .avi in the animations folder
#   ./convert_avi_to_mp4.sh file.avi [file2.avi]  # convert specific file(s)
#
# Env options:
#   FPS=15       frame rate of the output (default 15)
#   KEEP_AVI=1   keep the .avi after a successful conversion (default: delete it)

ANIM_DIR="/glade/work/nguyen/unc-shoals/animations"
FPS="${FPS:-15}"

# ffmpeg from the module system (outside MATLAB, so no library clash)
module load ffmpeg 2>/dev/null
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not found. Check 'module avail ffmpeg' and load the listed version."
    exit 1
fi

# Files to convert: arguments, or all .avi in the animations folder
if [ "$#" -gt 0 ]; then
    files=("$@")
else
    shopt -s nullglob
    files=("${ANIM_DIR}"/*.avi)
fi

if [ "${#files[@]}" -eq 0 ]; then
    echo "No .avi files found in ${ANIM_DIR}"
    exit 0
fi

for avi in "${files[@]}"; do
    [ -f "$avi" ] || { echo "Skip (not found): $avi"; continue; }
    mp4="${avi%.avi}.mp4"
    echo "Converting: $avi"
    # H.264 needs even width/height -> round down to even
    if ffmpeg -y -loglevel error -i "$avi" -c:v libx264 -pix_fmt yuv420p \
           -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -r "$FPS" "$mp4"; then
        echo "  -> $mp4"
        [ "${KEEP_AVI:-0}" = "1" ] || rm -f "$avi"
    else
        echo "  Conversion failed; kept $avi"
    fi
done

