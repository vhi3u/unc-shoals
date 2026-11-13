function video_writer_finish(vw, mp4_path, avi_path, fps)
% VIDEO_WRITER_FINISH  Close a writer and transcode the AVI fallback to MP4.
%
%   video_writer_finish(vw, mp4_path, avi_path, fps)
%
% avi_path comes from VIDEO_WRITER_OPEN: '' means MPEG-4 was written directly
% and there is nothing to do. Otherwise the AVI is converted to H.264 MP4 and
% deleted, leaving exactly the file the caller asked for.
%
% Needs ffmpeg ON PATH IN THE MATLAB PROCESS. The runners therefore have to
% 'module load ffmpeg' BEFORE starting MATLAB -- none of them did, which is
% why runs were leaving .avi behind with the conversion silently skipped.
% Motion-JPEG AVI is also roughly an order of magnitude larger than H.264,
% so the AVIs are not a drop-in substitute.

close(vw);

if isempty(avi_path)
    fprintf('Wrote MP4: %s\n', mp4_path);
    return;
end

if system('command -v ffmpeg >/dev/null 2>&1') ~= 0
    fprintf(['Wrote AVI: %s\n' ...
             'ffmpeg is not on PATH, so it was NOT converted to MP4. Either add\n' ...
             '    module load ffmpeg\n' ...
             'to the runner before it starts MATLAB, or convert afterwards with\n' ...
             '    ./tools/convert_avi_to_mp4.sh\n'], avi_path);
    return;
end

% MATLAB prepends its own, older libstdc++ to LD_LIBRARY_PATH, so a
% module-loaded ffmpeg called from inside MATLAB dies with "GLIBCXX_... not
% found". Drop MATLAB's own directories for the child process only.
ld  = strsplit(getenv('LD_LIBRARY_PATH'), ':');
ld  = ld(~contains(ld, matlabroot) & ~cellfun(@isempty, ld));
% H.264 requires even frame dimensions; MATLAB figures are frequently odd.
% -crf 16: x264's default is 23, tuned for camera footage. Scientific
% figures are the worst case for it -- large flat areas next to one-pixel
% text and thin contour lines, exactly what the rate control throws away
% first. 16 is visually lossless for this content and the files are small
% anyway at 51 frames. -preset slow buys better quality at the same CRF;
% encode time is seconds here.
%
% yuv420p is kept for player compatibility even though chroma subsampling
% costs some sharpness on coloured text. yuv444p would be sharper but will
% not open in QuickTime or in a lot of presentation software.
cmd = sprintf(['LD_LIBRARY_PATH="%s" ffmpeg -y -loglevel error -i "%s" ' ...
               '-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p ' ...
               '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -r %g "%s"'], ...
              strjoin(ld, ':'), avi_path, fps, mp4_path);

[st, out] = system(cmd);
if st == 0
    delete(avi_path);
    fprintf('Wrote MP4 (ffmpeg): %s\n', mp4_path);
else
    fprintf('ffmpeg conversion failed (%s).\nKept AVI: %s\n', strtrim(out), avi_path);
end
end
