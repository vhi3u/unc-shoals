function [vw, avi_path] = video_writer_open(mp4_path, fps)
% VIDEO_WRITER_OPEN  Open a VideoWriter for an MP4, falling back to AVI.
%
%   [vw, avi_path] = video_writer_open(mp4_path, fps)
%
% MATLAB's 'MPEG-4' profile is a Windows/macOS feature. On Linux -- which is
% every MATLAB on Casper -- requesting it raises
%
%   The specified profile is not valid.
%
% There is no MATLAB-side fix and no profile switch that produces H.264: the
% encoder is simply absent from the Linux build. So write Motion-JPEG AVI,
% which always works, and transcode after the last frame.
%
% avi_path is '' when MPEG-4 opened directly (Windows/macOS) and the AVI path
% otherwise. Pass it, unchanged, to VIDEO_WRITER_FINISH, which does the
% transcode and the reporting.
%
% Callers: vishoals.m, vishoals_clipped.m, vishoals_2panel.m. The fallback
% used to exist in vishoals.m only, so the other two threw on Casper rather
% than degrading.

avi_path = '';
try
    vw = VideoWriter(mp4_path, 'MPEG-4');
catch ME
    [out_d, base] = fileparts(mp4_path);
    avi_path = fullfile(out_d, [base '.avi']);
    fprintf('MPEG-4 not available (%s).\nWriting Motion-JPEG AVI instead: %s\n', ...
            ME.message, avi_path);
    vw = VideoWriter(avi_path, 'Motion JPEG AVI');
    vw.Quality = 95;
end
vw.FrameRate = fps;
open(vw);
end
