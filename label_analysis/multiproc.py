# %%
import logging
import sys
from pathlib import Path

import itk
import pandas as pd
import ray
import SimpleITK as sitk
from label_analysis.geometry import LabelMapGeometry
from label_analysis.geometry_itk import LabelMapGeometryITK
from label_analysis.helpers import *
from label_analysis.overlap import BatchScorerWorkerITK, BatchScorerWorkerPT
from label_analysis.radiomics_setup import *
from utilz.cprint import cprint
from utilz.helpers import *

itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(8)


def _concat_valid_frames(frames):
    valid_frames = []
    for frame in frames:
        if frame is None or not isinstance(frame, pd.DataFrame):
            continue
        if frame.empty:
            continue
        if frame.dropna(axis=1, how="all").empty:
            continue
        valid_frames.append(frame)

    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def _processing_error_row(gt_fn, err):
    return pd.DataFrame(
        [
            {
                "lm_filename": gt_fn,
                "processing_error": True,
                "error_type": type(err).__name__,
                "error_message": str(err),
            }
        ]
    )


def _process_labelmap_batch_itk(
    gt_fns,
    ignore_labels=None,
    dusting_threshold=0,
    img_fns=None,
    do_radiomics=True,
    params_fn=None,
):
    nbrhoods = []
    if ignore_labels is None:
        ignore_labels = []
    if isinstance(gt_fns, (str, Path)):
        gt_path = Path(gt_fns)
        gt_fns = list(gt_path.glob("*")) if gt_path.is_dir() else [gt_path]
    elif not isinstance(gt_fns, list):
        gt_fns = list(gt_fns)
    if img_fns is None:
        img_fns = []
    elif isinstance(img_fns, (str, Path)):
        img_path = Path(img_fns)
        img_fns = list(img_path.glob("*")) if img_path.is_dir() else [img_path]
    elif not isinstance(img_fns, list):
        img_fns = list(img_fns)
    if len(img_fns) == 0 and do_radiomics:
        cprint(
            "Warning: no img_files given. Radiomics will not be computed",
            color="red",
            bold=True,
        )
        do_radiomics = False
    for gt_fn in gt_fns:
        if len(img_fns) > 0:
            img_fn = find_matching_fn(
                gt_fn, img_fns, tags=["case_id"], allow_multiple_matches=False
            )
            img = sitk.ReadImage(img_fn)
        else:
            img = None
        try:
            L = LabelMapGeometryITK(li=gt_fn, ignore_labels=ignore_labels, img=img)
            if L.is_empty() is False:
                L.dust(dusting_threshold=dusting_threshold)
            if do_radiomics is True and L.is_empty() is False:
                L.radiomics(params_fn)
            L.nbrhoods["lm_filename"] = gt_fn
            L.nbrhoods["processing_error"] = False
            L.nbrhoods["error_type"] = None
            L.nbrhoods["error_message"] = None
            nbrhoods.append(L.nbrhoods)
        except Exception as e:
            logging.exception("Failed processing labelmap file: %s", gt_fn)
            nbrhoods.append(_processing_error_row(gt_fn, e))
    return _concat_valid_frames(nbrhoods)


@ray.remote(num_cpus=4)
class LabelMapGeometryRay:
    def __init__(self):
        pass

    def process(
        self,
        gt_fns,
        ignore_labels=None,
        dusting_threshold=0,
        img_fns=None,
        do_radiomics=True,
        params_fn=None,
    ):
        nbrhoods = []
        if ignore_labels is None:
            ignore_labels = []
        if isinstance(gt_fns, (str, Path)):
            gt_path = Path(gt_fns)
            gt_fns = list(gt_path.glob("*")) if gt_path.is_dir() else [gt_path]
        elif not isinstance(gt_fns, list):
            gt_fns = list(gt_fns)
        if img_fns is None:
            img_fns = []
        elif isinstance(img_fns, (str, Path)):
            img_path = Path(img_fns)
            img_fns = list(img_path.glob("*")) if img_path.is_dir() else [img_path]
        elif not isinstance(img_fns, list):
            img_fns = list(img_fns)
        if len(img_fns) == 0 and do_radiomics:
            cprint(
                "Warning: no img_files given. Radiomics will not be computed",
                color="red",
                bold=True,
            )
            do_radiomics = False
        for gt_fn in gt_fns:
            try:
                if len(img_fns) > 0:
                    img_fn = find_matching_fn(
                        gt_fn, img_fns, tags=["case_id"], allow_multiple_matches=False
                    )
                    img = sitk.ReadImage(img_fn)
                else:
                    img = None
                L = LabelMapGeometry(li=gt_fn, ignore_labels=ignore_labels, img=img)
                if do_radiomics == True and L.is_empty() == False:
                    L.dust(dusting_threshold=dusting_threshold)
                    L.radiomics(params_fn)
                L.nbrhoods["lm_filename"] = gt_fn
                L.nbrhoods["processing_error"] = False
                L.nbrhoods["error_type"] = None
                L.nbrhoods["error_message"] = None
                nbrhoods.append(L.nbrhoods)
            except Exception as e:
                logging.exception("Failed processing labelmap file: %s", gt_fn)
                nbrhoods.append(_processing_error_row(gt_fn, e))

        return _concat_valid_frames(nbrhoods)


@ray.remote(num_cpus=4)
class LabelMapGeometryRayITK:
    def __init__(self):
        pass

    def process(
        self,
        gt_fns,
        ignore_labels=None,
        dusting_threshold=0,
        img_fns=None,
        do_radiomics=True,
        params_fn=None,
    ):
        return _process_labelmap_batch_itk(
            gt_fns=gt_fns,
            ignore_labels=ignore_labels,
            dusting_threshold=dusting_threshold,
            img_fns=img_fns,
            do_radiomics=do_radiomics,
            params_fn=params_fn,
        )


@ray.remote(num_cpus=4)
class BatchScorerWorkerRay:
    def __init__(self, actor_id):
        self.actor_id = actor_id

    def process(
        self,
        gt_fns: Union[Path, list],
        preds_fldr: Path | list[Path],
        ignore_labels_gt: list,
        ignore_labels_pred: list,
        imgs_fldr: Path = None,
        partial_df: pd.DataFrame = None,
        exclude_fns=[],
        output_fldr=None,
        do_radiomics=False,
        dusting_threshold=1,
        debug=False,
        batch_scorer_cls=BatchScorerWorkerITK,
    ):
        print("process {} ".format(self.actor_id))
        self.B = batch_scorer_cls(
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


class BatchScorerRayITK:
    batch_scorer_cls = BatchScorerWorkerITK

    def __init__(
        self,
        gt_fns: list[Path],
        preds_fldr: Path | list[Path],
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
        if isinstance(preds_fldr, Path):
            if not preds_fldr.exists():
                raise FileNotFoundError(
                    f"Predictions folder does not exist: {preds_fldr}"
                )
            preds_root = preds_fldr
        else:
            if not preds_fldr:
                raise ValueError("preds_fldr is empty")
            preds_root = preds_fldr[0].parent
        if output_fldr is None:
            output_fldr = preds_root / "results"
        if exclude_fns is None:
            exclude_fns = []

        output_fldr.mkdir(parents=True, exist_ok=True)

        self.gt_fns = gt_fns
        self.preds_fldr = preds_fldr
        self.ignore_labels_gt = ignore_labels_gt
        self.ignore_labels_pred = ignore_labels_pred
        self.output_fldr = output_fldr
        self.imgs_fldr = imgs_fldr
        self.partial_df = partial_df
        self.exclude_fns = exclude_fns
        self.do_radiomics = do_radiomics
        self.dusting_threshold = dusting_threshold
        self.debug = debug
        self.n_actors = min(len(gt_fns), n_actors)
        self.gt_fns_chunks = [gt_fns[i :: self.n_actors] for i in range(self.n_actors)]
        self.actors = [
            BatchScorerWorkerRay.remote(actor_id) for actor_id in range(self.n_actors)
        ]

    def process(self):
        futures = []
        for actor, gt_fns_chunk in zip(self.actors, self.gt_fns_chunks):
            futures.append(
                actor.process.remote(
                    gt_fns=gt_fns_chunk,
                    preds_fldr=self.preds_fldr,
                    ignore_labels_gt=self.ignore_labels_gt,
                    ignore_labels_pred=self.ignore_labels_pred,
                    imgs_fldr=self.imgs_fldr,
                    partial_df=self.partial_df,
                    exclude_fns=self.exclude_fns,
                    output_fldr=self.output_fldr,
                    do_radiomics=self.do_radiomics,
                    dusting_threshold=self.dusting_threshold,
                    debug=self.debug,
                    batch_scorer_cls=self.batch_scorer_cls,
                )
            )

        results = ray.get(futures)
        df = pd.concat(results, ignore_index=True)
        output_fn = (
            self.output_fldr
            / f"{self.output_fldr.name}_thresh{self.dusting_threshold}mm_all.xlsx"
        )
        df.to_excel(output_fn, index=False)
        print(f"Saved merged results to {output_fn}")
        return df


class BatchScorerRayPT(BatchScorerRayITK):
    batch_scorer_cls = BatchScorerWorkerPT


def _score_preds_folder(
    gt_fns: list[Path],
    preds_fldr: Path | list[Path],
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
    scorer_ray_cls=BatchScorerRayITK,
):
    scorer = scorer_ray_cls(
        gt_fns=gt_fns,
        preds_fldr=preds_fldr,
        ignore_labels_gt=ignore_labels_gt,
        ignore_labels_pred=ignore_labels_pred,
        output_fldr=output_fldr,
        imgs_fldr=imgs_fldr,
        partial_df=partial_df,
        exclude_fns=exclude_fns,
        do_radiomics=do_radiomics,
        dusting_threshold=dusting_threshold,
        debug=debug,
        n_actors=n_actors,
    )
    return scorer.process()


def score_preds_folder_itk(
    gt_fns: list[Path],
    preds_fldr: Path | list[Path],
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
    return _score_preds_folder(
        gt_fns=gt_fns,
        preds_fldr=preds_fldr,
        ignore_labels_gt=ignore_labels_gt,
        ignore_labels_pred=ignore_labels_pred,
        output_fldr=output_fldr,
        imgs_fldr=imgs_fldr,
        partial_df=partial_df,
        exclude_fns=exclude_fns,
        do_radiomics=do_radiomics,
        dusting_threshold=dusting_threshold,
        debug=debug,
        n_actors=n_actors,
        scorer_ray_cls=BatchScorerRayITK,
    )


def score_preds_folder_pt(
    gt_fns: list[Path],
    preds_fldr: Path | list[Path],
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
    return _score_preds_folder(
        gt_fns=gt_fns,
        preds_fldr=preds_fldr,
        ignore_labels_gt=ignore_labels_gt,
        ignore_labels_pred=ignore_labels_pred,
        output_fldr=output_fldr,
        imgs_fldr=imgs_fldr,
        partial_df=partial_df,
        exclude_fns=exclude_fns,
        do_radiomics=do_radiomics,
        dusting_threshold=dusting_threshold,
        debug=debug,
        n_actors=n_actors,
        scorer_ray_cls=BatchScorerRayPT,
    )


# %%
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == "__main__":
    from fran.data.dataregistry import DS
    from utilz.helpers import info_from_filename

    if not ray.is_initialized():
        ray.init()

    preds_fldr = Path("/s/fran_storage/predictions/kits23/KITS23-SIRIG")
    pred_fns = sorted(preds_fldr.glob("kits23*"))
    pred_case_ids = {
        info_from_filename(fn.name, full_caseid=True)["case_id"] for fn in pred_fns
    }
    gt_fldr = DS.kits23.folder / "lms"
    gt_fns = [
        fn
        for fn in sorted(gt_fldr.glob("*"))
        if info_from_filename(fn.name, full_caseid=True)["case_id"] in pred_case_ids
    ]

# %%
    df = score_preds_folder_itk(
        gt_fns=gt_fns,
        preds_fldr=pred_fns,
        ignore_labels_gt=[1],
        ignore_labels_pred=[1],
        output_fldr=preds_fldr / "results2",
        do_radiomics=False,
        dusting_threshold=1,
        debug=False,
        n_actors=8,
    )
# %%
# SECTION:-------------------- PT SCORING--------------------------------------------------------------------------------------
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

    dfbl_tw = dfbl.merge(on="case_id", how="outer", right=dft2, suffixes=("_bl", "_tw"))

    dfbl_tw.to_csv("comp_all.csv")
    mn = dfbl_tw["dsc_overall_bl"].median()
    mn2 = dfbl_tw["dsc_overall_tw"].median()
# %%
    table = (
        df.dropna(subset=["dsc"])
        .groupby("case_id")["dsc_overall"]
        .unique()
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
# %%
    df2.to_csv("bl.csv")
# %%
