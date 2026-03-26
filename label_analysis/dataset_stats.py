import os
import errno
import sys

from utilz.cprint import cprint
from utilz.fileio import maybe_makedirs

from label_analysis.multiproc import (LabelMapGeometryRayITK,
                                      _concat_valid_frames)
from label_analysis.overlap import chunks

sys.path += ["/home/ub/code"]
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from utilz.helpers import *

from label_analysis.helpers import *


def split_info(full_filename_posix):
    fn_name = full_filename_posix.name
    case_id = info_from_filename(fn_name, True)["case_id"]
    return case_id


def lms_folder_statistics(
    input_folder, output_folder=None, ignore_labels=[], dusting_threshold=0
):
    if ignore_labels:
        assert isinstance(ignore_labels, list), "ignore_labels must be a list"
    input_folder = Path(input_folder)
    if output_folder is None:
        output_folder = input_folder.parent / "label_analysis"
    output_folder = Path(output_folder)
    out_fn = output_folder / ("lesion_stats.csv")
    if out_fn.exists():
        cprint(f"Skipping: {out_fn} already exists", color="red", bold=True)
        return pd.read_csv(out_fn)
    fns_pt = list(input_folder.glob("*"))
    if len(fns_pt) == 0:
        raise ValueError(f"No files found in input folder: {input_folder}")

    n_actors = 8
    fns_chunks = list(chunks(fns_pt, n_actors))
    fns_chunks = [chunk for chunk in fns_chunks if len(chunk) > 0]

    actors = [LabelMapGeometryRayITK.remote() for _ in range(len(fns_chunks))]
    do_radiomics = False
    params_fn = None
    futures = []
    for actor, fns_chunk in zip(actors, fns_chunks):
        res = actor.process.remote(
            fns_chunk,
            ignore_labels,
            dusting_threshold,
            img_fns=None,
            do_radiomics=do_radiomics,
            params_fn=params_fn,
        )
        futures.append(res)

    parts = ray.get(futures)
    resdf = _concat_valid_frames(parts)

    if "lm_filename" in resdf.columns:
        resdf["case_id"] = resdf["lm_filename"].apply(split_info)

    has_actor_errors = (
        "processing_error" in resdf.columns
        and resdf["processing_error"].fillna(False).astype(bool).any()
    )
    if has_actor_errors:
        err_rows = resdf[resdf["processing_error"].fillna(False).astype(bool)]
        err_counts = err_rows["error_type"].value_counts(dropna=False).to_dict()
        sample_msgs = err_rows["error_message"].dropna().astype(str).head(3).tolist()
        pipeline_msg = (
            "Stopped: one or more actor tasks failed. "
            f"error_type_counts={err_counts}. "
            f"sample_error_messages={sample_msgs}"
        )
        resdf["pipeline_error_message"] = pipeline_msg

    maybe_makedirs([output_folder])
    cprint(f"Saving to {out_fn}", color="green")
    resdf.to_csv(out_fn, index=False)

    return resdf


def plot_lesion_volume_distributions(
    df_input: Path | str | pd.DataFrame, output_folder
):
    if isinstance(df_input, str | Path):
        df = pd.read_csv(df_input)
    else:
        df = df_input

    os.makedirs(output_folder, exist_ok=True)

    def _pick_first_existing(
        columns: list[str], candidates: list[str], kind: str
    ) -> str:
        for col in candidates:
            if col in columns:
                return col
        raise KeyError(
            f"Missing {kind} column. Expected one of {candidates}; available columns: {list(columns)}"
        )

    vol_col = _pick_first_existing(
        columns=list(df.columns),
        candidates=["volume_cc", "volume"],
        kind="volume",
    )
    len_col = _pick_first_existing(
        columns=list(df.columns),
        candidates=["major_axis", "length"],
        kind="length",
    )

    case_ids = sorted(df["case_id"].unique())

    chunk = 50
    n_chunks = math.ceil(len(case_ids) / chunk)
    counts = df.groupby("case_id").size()
    # order cases by decreasing lesion count
    ordered_cases = counts.sort_values(ascending=False).index.tolist()

    chunk_size = 50
    n_chunks = math.ceil(len(ordered_cases) / chunk_size)

    for i in range(n_chunks):

        out_fn = os.path.join(
            output_folder, f"volume_length_dualaxis_chunk_{i+1:02d}.png"
        )
        if os.path.exists(out_fn):
            cprint(f"Skipping: {out_fn} already exists", color="red", bold=True)
            continue
        start = i * chunk_size
        end = start + chunk_size
        chunk_cases = ordered_cases[start:end]
        d = df[df["case_id"].isin(chunk_cases)].copy()

        vol_data = [d.loc[d["case_id"] == case, vol_col].values for case in chunk_cases]
        len_data = [d.loc[d["case_id"] == case, len_col].values for case in chunk_cases]

        x = np.arange(len(chunk_cases))
        pos_vol = x - 0.18
        pos_len = x + 0.18

        fig, ax1 = plt.subplots(figsize=(24, 7))
        ax2 = ax1.twinx()

        bp1 = ax1.boxplot(
            vol_data,
            positions=pos_vol,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
        )

        bp2 = ax2.boxplot(
            len_data,
            positions=pos_len,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
        )

        # colors
        for b in bp1["boxes"]:
            b.set_facecolor("tab:blue")
            b.set_alpha(0.5)

        for b in bp2["boxes"]:
            b.set_facecolor("tab:orange")
            b.set_alpha(0.5)

        # overlay dots
        rng = np.random.default_rng(42)

        for xv, vals in zip(pos_vol, vol_data):
            jitter = rng.uniform(-0.05, 0.05, size=len(vals))
            ax1.scatter(
                np.full(len(vals), xv) + jitter,
                vals,
                s=8,
                alpha=0.5,
                color="tab:blue",
            )

        for xl, vals in zip(pos_len, len_data):
            jitter = rng.uniform(-0.05, 0.05, size=len(vals))
            ax2.scatter(
                np.full(len(vals), xl) + jitter,
                vals,
                s=8,
                alpha=0.5,
                color="tab:orange",
            )

        # lesion counts at top
        ymax1 = max(np.max(v) for v in vol_data if len(v) > 0)
        for xi, case in zip(x, chunk_cases):
            ax1.text(
                xi,
                ymax1 * 1.03,
                str(int(counts[case])),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax1.set_xlim(-0.8, len(chunk_cases) - 0.2)
        ax1.set_ylim(0, ymax1 * 1.12)

        ymax2 = max(np.max(v) for v in len_data if len(v) > 0)
        ax2.set_ylim(0, ymax2 * 1.12)

        ax1.set_xticks(x)
        ax1.set_xticklabels(chunk_cases, rotation=90)

        ax1.set_xlabel("case_id")
        ax1.set_ylabel(vol_col, color="tab:blue")
        ax2.set_ylabel(len_col, color="tab:orange")

        ax1.tick_params(axis="y", colors="tab:blue")
        ax2.tick_params(axis="y", colors="tab:orange")

        ax1.set_title("Volume and length distributions by case")

        plt.tight_layout()
        plt.savefig(out_fn, dpi=300)
        plt.close()


def end2end_lms_stats_and_plots(
    lis_folder, output_folder=None, ignore_labels=None, dusting_threshold=0
):
    lis_folder = Path(lis_folder)
    output_folder = (
        Path(output_folder) if output_folder else lis_folder.parent / "dataset_stats"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    ignore_labels = [] if ignore_labels is None else ignore_labels
    df = lms_folder_statistics(
        input_folder=lis_folder,
        output_folder=output_folder,
        ignore_labels=ignore_labels,
        dusting_threshold=dusting_threshold,
    )
    has_actor_errors = (
        "processing_error" in df.columns
        and df["processing_error"].fillna(False).astype(bool).any()
    )
    if has_actor_errors:
        cprint(
            "Skipping plotting: actor processing errors detected. "
            "Inspect `processing_error`, `error_type`, `error_message`, and `pipeline_error_message`.",
            color="red",
            bold=True,
        )
        return df, output_folder / "lesion_stats.csv"

    plot_lesion_volume_distributions(df_input=df, output_folder=output_folder)
    return df, output_folder / "lesion_stats.csv"


# %%
if __name__ == "__main__":

    from fran.managers.project import DS
    fldr_pt = Path(
        "/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex050/lms"
    )
    fldr_lidc = Path("/media/UB/datasets/lidc/lms")
    output_fldr = fldr_pt.parent / ("label_analysis")
    output_lidc = fldr_lidc.parent / ("label_analysis")
    df = lms_folder_statistics(fldr_lidc, None, dusting_threshold=1, ignore_labels=[])
# %%
    fldrs=[]
    ignore_labels_lidc = []
    main_fldr = Path("/r/datasets/preprocessed/lidc")
    for fldr_name in main_fldr.rglob("*"):
        if "lms" in fldr_name.name and fldr_name.is_dir():
            fldrs.append(fldr_name)

# %%
    for fldr in fldrs:
        end2end_lms_stats_and_plots(
            lis_folder=fldr,
            ignore_labels=ignore_labels_lidc,
            dusting_threshold=1,)
# %%
# SECTION:-------------------- end to end pipeline-------------------------------------------------------------------------------------- <CR>
    
# %%
# %%
# SECTION:-------------------- LIDC-------------------------------------------------------------------------------------- <CR>

    if not ray.is_initialized():
        ray.init()
    DS.lidc2
    end2end_lms_stats_and_plots(
        lis_folder="/media/UB/datasets/lidc2/lms_filled",
        ignore_labels=[1],
        dusting_threshold=1,
        output_folder="/media/UB/datasets/lidc2/dataset_stats_filled",
    )
# %%
    df = lms_folder_statistics(
        fldr_pt, output_lidc, dusting_threshold=1, ignore_labels=[]
    )
# %%
    plot_lesion_volume_distributions(df, output_lidc)
    import seaborn as sns

# %%
    df_fn = "/media/UB/datasets/kits23/dataset_stats/lesion_stats.csv"

    df = pd.read_csv(df_fn)
    ser = df.groupby("case_id").size()
    ser.sort_values(ascending=False)
    sns.histplot(ser)
# %%

    csv_fn = Path(
        "/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000/label_stats/lesion_stats.csv"
    )
    csv_fn = "/media/UB/datasets/kits23/label_analysis/lesion_stats.csv"
    df = pd.read_csv(csv_fn)
    counts = df.groupby("case_id").size()
    counts = counts.sort_values(ascending=False)

    o_counts = counts.sort_values(ascending=False).index.tolist()
# %%
