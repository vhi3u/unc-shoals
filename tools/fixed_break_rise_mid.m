function rise_mid = fixed_break_rise_mid(runs, min_gap)
% rise_mid = fixed_break_rise_mid(runs, min_gap)
%
% ONE shelf-break position for a whole sweep: the smallest rise_mid at which
% every run in `runs` (a struct array of shoal_bathymetry parameters) keeps
% at least min_gap [m, default 10 km] of flat shelf between the end of its
% shoal's offshore taper and the start of the shelf break.
%
% The break is fixed per sweep, not per run, so the jet and the analysis box
% (both fixed in x) sit over the same topography in every run. The run with
% the shoal reaching furthest offshore sets it; the others get more room,
% which shoal_bathymetry measures as geom.shelf_gap.
%
% The break starts (background 1 m below Zsh) at
%   rise_mid + rise_width * atanh(2 / (Zsh - h_deep) - 1),
% so the requirement for each run is
%   rise_mid >= taper_end + min_gap - rise_width * atanh(2 / (Zsh - h_deep) - 1).

if nargin < 2 || isempty(min_gap), min_gap = 10e3; end
need = -Inf;
for k = 1:numel(runs)
    p = runs(k);
    [~, g] = shoal_bathymetry(0, 0, p);              % taper_end is set by the shoal only
    q = defaults_(p);
    if isnan(g.taper_end), continue; end              % no shoal: nothing to keep clear
    drop = q.Zsh - q.h_deep;
    need = max(need, g.taper_end + min_gap - q.rise_width * atanh(2 / drop - 1));
end
rise_mid = need;
end

function p = defaults_(p)
d = struct('Zsh', -25, 'h_deep', -50, 'rise_width', 5e3);
if isfield(p, 'shelf_depth') && ~isfield(p, 'Zsh'), p.Zsh = p.shelf_depth; end
for f = fieldnames(d)', if ~isfield(p, f{1}) || isempty(p.(f{1})), p.(f{1}) = d.(f{1}); end, end
end
