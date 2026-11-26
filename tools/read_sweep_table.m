function runs = read_sweep_table(toml_file)
% runs = read_sweep_table(toml_file)
%
% The [[runs]] entries of a sweep table (src/sweep/params_*.toml) as a struct
% array, one element per run, one field per key. Handles what these tables
% use: `key = number`, `key = "string"`, `key = true/false`, and # comments.

txt = splitlines(fileread(toml_file));
runs = struct([]);
cur = [];
for i = 1:numel(txt)
    ln = strtrim(txt{i});
    if isempty(ln) || startsWith(ln, '#'), continue; end
    if strcmp(ln, '[[runs]]')
        if ~isempty(cur), runs = append_(runs, cur); end
        cur = struct();
        continue;
    end
    if isempty(cur) || ~contains(ln, '='), continue; end
    [key, val] = strtok(ln, '=');
    key = strtrim(key);  val = strtrim(val(2:end));
    if startsWith(val, '"')                         % string (may be followed by a comment)
        q = find(val == '"');
        val = val(q(1)+1 : q(2)-1);
    else
        c = strfind(val, '#');  if ~isempty(c), val = strtrim(val(1:c(1)-1)); end
        if strcmpi(val, 'true'), val = true;
        elseif strcmpi(val, 'false'), val = false;
        else, val = str2double(val);
        end
    end
    cur.(key) = val;
end
if ~isempty(cur), runs = append_(runs, cur); end
end

function runs = append_(runs, cur)
% struct arrays need identical fields: fill missing ones with []
if isempty(runs), runs = cur; return; end
for f = setdiff(fieldnames(cur), fieldnames(runs))', [runs.(f{1})] = deal([]); end
for f = setdiff(fieldnames(runs), fieldnames(cur))', cur.(f{1}) = []; end
runs(end+1) = orderfields(cur, runs(1));
end
