# tools/ — MATLAB analysis and plotting

**This directory is the single source of truth for all visualization code.**

It was created in Nov 2026 by merging two copies that had drifted apart:

- `~/Documents/v_ocean/vnmain/matlabvis` (laptop) — where `.m` analysis code was authored
- `/glade/work/nguyen/unc-shoals/tools` (Casper) — where `run_*.sh` launchers were written

Neither contained the other. The laptop had 12 of 50 runners; the cluster was
missing the most recent `.m` edits. Reconciling them took a full checksum diff.

## Workflow — do this

```
edit in  unc-shoals/tools/   →   git commit   →   git push
                                                      ↓
                        on Casper:  cd /glade/work/nguyen/unc-shoals && git pull
```

## Do NOT do this

**Never `scp` into `tools/` on Casper.** That is what caused the drift. Anything
scp'd is invisible to git, gets clobbered by the next `git pull`, and has to be
hand-reconciled. Edit locally, commit, pull.

The old `vnmain/matlabvis` directory is superseded. Do not edit there.

## Running a script on Casper

```bash
cd /glade/work/nguyen/unc-shoals/tools
./run_vishoals.sh [cluster_name]
```

Each `run_*.sh` purges modules, loads `ncarenv-basic/23.10`, `ncarenv/23.10`
and `matlab/R2024a`, then calls its `.m` entry point.

## Third-party toolboxes — not in git

`gsw_matlab_v3_06_16/`, `cmocean/` and `pmkmp/` live only on Casper and are
gitignored. A clone anywhere else must install them before the runners work.

## Known issues

- **Stale cluster defaults.** 19 runners still default to `cluster13_111_117`,
  `cluster10_81_89` or `cluster_Cnnie_1_15`. The current sweep is
  `cluster_Fnnie_1_19` (= `src/sweep/params_HsLs.toml`, runs 1-19). Running one
  of those 19 with no argument silently plots months-old data. Always pass the
  cluster name explicitly until the defaults are swept.
- **9 legacy `.m` files** (`make_*_gif.m`, `visxy.m`, `visxz.m`, `zavg.m`) write
  to hardcoded `/Users/victornguyen/...` paths and cannot run on Casper. They
  predate `vishoals.m` and are probably safe to delete.
- Eight runners previously had `addpath('/glade/work/nguyen/vnmain/matlabvis')`,
  a path that never existed on Casper. Removed; the addpath was always a no-op.

## Paper figures → scripts

| Figure | Script |
|---|---|
| animation | `vishoals.m` / `run_vishoals.sh` |
| still frame (last day) | `vishoals_still.m`, `vishoals_still_bulk.m` |
| overturning + recirculation | `vis_overturning_recirc.m` |
| cross-shelf transport | `vis_offshore_3panel.m` |
| sweep scatter (Hs, Ls) | `plot_full_scatter_sweeps.m` |
| surface vorticity vs wind | `vishoals_vort_wind.m` |
| surface EKE sweeps (+log) | `vis_eke_sweeps.m` |
| model vs theory | `plot_theory_vs_model.m`, `calc_Q_EKE.m` |
