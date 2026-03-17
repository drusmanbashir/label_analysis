from pathlib import Path

import pandas as pd
import ray

from label_analysis.overlap import BatchScorerPT



@ray.remote(num_cpus=4)
class BatchScorerRayPT:
    def __init__(self, actor_id):
        self.actor_id = actor_id

    def process(
        self,
        gt_fns: Path | list,
        preds_fldr: Path,
        ignore_labels_gt: list,
        ignore_labels_pred: list,
        imgs_fldr: Path = None,
        partial_df: pd.DataFrame = None,
        exclude_fns=[],
        output_fldr=None,
        do_radiomics=False,
        dusting_threshold=1,
        debug=False,
    ):
        print("process {} ".format(self.actor_id))
        self.B = BatchScorerPT(
            output_suffix=self.actor_id,
            gt_fns=gt_fns,
            preds_fldr=preds_fldr,
            ignore_labels_gt=ignore_labels_gt,
            ignore_labels_pred=ignore_labels_pred,
            imgs_fldr=imgs_fldr,
            partial_df=partial_df,
            exclude_fns=exclude_fns,
            do_radiomics=do_radiomics,
            dusting_threshold=dusting_threshold,
            debug=debug,
        )
        return self.B.process()


def score_preds_folder_pt(
    gt_fns: list[Path],
    preds_fldr: Path,
    ignore_labels_gt: list,
    ignore_labels_pred: list,
    output_fldr: Path | None = None,
    imgs_fldr: Path | None = None,
    partial_df: pd.DataFrame | None = None,
    exclude_fns: list | None = None,
    do_radiomics: bool = False,
    dusting_threshold: int = 1,
    debug: bool = False,
    n_actors: int = 8,
):
    if not gt_fns:
        raise ValueError("gt_fns is empty")
    if not preds_fldr.exists():
        raise FileNotFoundError(f"Predictions folder does not exist: {preds_fldr}")
    if output_fldr is None:
        output_fldr = preds_fldr / "results"
    if exclude_fns is None:
        exclude_fns = []

    output_fldr.mkdir(parents=True, exist_ok=True)

    n_actors = min(len(gt_fns), n_actors)
    gt_fns_chunks = [gt_fns[i::n_actors] for i in range(n_actors)]

    actors = [BatchScorerRayPT.remote(actor_id) for actor_id in range(n_actors)]
    futures = []
    for actor, gt_fns_chunk in zip(actors, gt_fns_chunks):
        futures.append(
            actor.process.remote(
                gt_fns=gt_fns_chunk,
                preds_fldr=preds_fldr,
                ignore_labels_gt=ignore_labels_gt,
                ignore_labels_pred=ignore_labels_pred,
                imgs_fldr=imgs_fldr,
                partial_df=partial_df,
                exclude_fns=exclude_fns,
                output_fldr=output_fldr,
                do_radiomics=do_radiomics,
                dusting_threshold=dusting_threshold,
                debug=debug,
            )
        )

    results = ray.get(futures)
    df = pd.concat(results, ignore_index=True)
    output_fn = output_fldr / f"{output_fldr.name}_thresh{dusting_threshold}mm_all.xlsx"
    df.to_excel(output_fn, index=False)
    print(f"Saved merged results to {output_fn}")
    return df


if __name__ == "__main__":
# %%
# SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR>

    from fran.managers.project import Project
    from utilz.helpers import info_from_filename

    P = Project("kits2")

    _, val = P.get_train_val_case_ids(0)
    gt_fldr = Path(
        "/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms"
    )
    gt_fns = sorted(gt_fldr.glob("*.pt"))
    val_fns = [
        fn
        for fn in gt_fns
        if info_from_filename(fn.name, full_caseid=True)["case_id"] in val
    ]
    len(val_fns)
    ignore_labels_gt = [1]
    ignore_labels_pred = [1]
    # preds_fldr = Path("/s/fran_storage/predictions/kits/KITS-n7")
    preds_fldr = Path("/s/fran_storage/predictions/kits2/KITS2-bk")
    # gt_fns2 = gt_fns[:5]+ [Path("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms/kits23_00030.pt")]

    pred_fns = [
        fn
        for fn in preds_fldr.glob("*")
        if info_from_filename(fn.name, full_caseid=True)["case_id"] in val
    ]
    print(len(pred_fns))

# %%
    gt_fns = val_fns
    df = score_preds_folder_pt(
        gt_fns=gt_fns,
        preds_fldr=preds_fldr,
        ignore_labels_gt=ignore_labels_gt,
        ignore_labels_pred=ignore_labels_pred,
        output_fldr=preds_fldr / "results",
        do_radiomics=False,
        dusting_threshold=1,
        debug=False,
    )

# %%
    df_TW = pd.read_excel(
        "/s/fran_storage/predictions/kits/KITS-TW/results/results_thresh1mm_results1.xlsx"
    )
    df2 = pd.read_excel(
        "/s/fran_storage/predictions/kits/KITS-bl/results/results_thresh1mm_all.xlsx"
    )
    df.to_csv("bl_all.csv")
# %%
    df_TW.columns
    len(df_TW.groupby("case_id"))

    df.loc[df["case_id"] == "kits23_00030", "dsc"]
    df2.loc[df2["case_id"] == "kits23_00030", "dsc"].unique()

    df2.dropna(subset=["dsc"], inplace=True)
    df2["dsc"].unique()

    tbl = (
        (df2.dropna(subset=["dsc"]))
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
# %%
    dfbl = df[["case_id", "dsc_overall"]].drop_duplicates("case_id")
    dft2 = df_TW[["case_id", "dsc_overall"]].drop_duplicates("case_id")

    dfbl_tw=dfbl.merge(on="case_id", how="outer", right=dft2, suffixes=("_bl", "_tw"))

    dfbl_tw.to_csv("comp_all.csv")
    mn = dfbl_tw["dsc_overall_bl"].median()
    mn2 = dfbl_tw["dsc_overall_tw"].median()
# %%
    table = (
        df.dropna(subset=["dsc"])
        .groupby("case_id")["dsc_overall"].unique()
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
# %%
    df2.to_csv("bl.csv")
# %%
