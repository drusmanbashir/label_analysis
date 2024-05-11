# %%
from label_analysis.overlap import ScorerAdvanced, ScorerFiles
from pathlib import Path
from label_analysis.utils import is_sitk_file
import pandas as pd
import numpy as np
from label_analysis.helpers import inds_to_labels
import itertools as il

from fran.utils.string import find_file
# %%

def fk_generator(start=0):
    key = start
    while True:
        yield key
        key+=1

# %%

class ScorerFiles2(ScorerAdvanced):
    def corresponding_gt_inds(self,pred_inds):
            gt_dsc_gps = []
            for ind in pred_inds:
                if hasattr(ind,'item'):
                    ind= ind.item()
                gp = np.nonzero(self.dsc[:,ind])
                gp = set(gp[0])
                gt_dsc_gps.append(gp)
            return gt_dsc_gps


    def dsc_gp_remapping(self,dsc_gps):
        remapping = {}
        dest_labels=[]
        for gp in dsc_gps:
            gp = inds_to_labels(gp)
            main_lab = gp[0]
            dest_labels.append(main_lab)
            maps = {lab:int(main_lab) for lab in gp}
            remapping.update(maps)
        return remapping,dest_labels

    def recompute_overlap_perlesion(self):

        row_counts = np.count_nonzero(self.dsc, 1)
        col_counts = np.count_nonzero(self.dsc, 0)
        
        pred_inds_m21=np.argwhere(col_counts>1).flatten().tolist()
        pred_inds_12x = np.argwhere(col_counts== 1).flatten().tolist()
        gt_inds_12m = np.argwhere(row_counts>1).flatten().tolist()

        fk_gen = fk_generator(0)
        self.dsc_single=[]
        fks_121 , pred_inds_121,gt_inds_121 = [],[], []
        for pred_ind in pred_inds_12x:
            # pred_ind = pred_inds_x21[ind]
            row_ind = np.argwhere(self.dsc[:,pred_ind]>0)
            if np.count_nonzero(self.dsc[row_ind,:])==1:
                ind_pair = {'gt_ind':row_ind.item(), 'pred_ind':pred_ind}
                pred_ind_121 = pred_ind
                gt_ind_121 =row_ind.item()
                gt_inds_121.append(row_ind.item())
                pred_inds_121.append(pred_ind)
                self.dsc_single.append(self.dsc[gt_ind_121,pred_ind_121])
                fks_121.append(next(fk_gen))

        gt_inds_m21 = self.corresponding_gt_inds(pred_inds_m21)
        inds = np.tril_indices(len(gt_inds_m21),-1)
        keep_inds=[True]*len(gt_inds_m21)
        gt_supersets = []
        for x,y in zip(*inds):
            set1 = gt_inds_m21[x]
            set2  = gt_inds_m21[y]
            if len(set1.intersection(set2))>0:
                    keep_inds[x]=False
                    keep_inds[y]=False
                    gt_supersets.append(set1.union(set2))
        self.gt_inds_m2m = list(il.compress(gt_inds_m21,keep_inds))  + gt_supersets

        pred_inds_m2m ,fks_m2m= [],[]
        # gt_inds = gt_inds_m2m[0]
        for gt_inds in self.gt_inds_m2m:
            gt_inds = list(gt_inds)
            pred_inds = set(np.argwhere(self.dsc[gt_inds,:])[:,1])
            pred_inds_m2m.append(pred_inds)
            fks_m2m.append(next(fk_gen))


        self.gt_labs_121 = inds_to_labels(gt_inds_121)
        pred_labs_121 = inds_to_labels(pred_inds_121)


        gt_remaps, self.gt_labs_m2m = self.dsc_gp_remapping(self.gt_inds_m2m)
        pred_remaps, pred_labs_m2m = self.dsc_gp_remapping(pred_inds_m2m)

        self.LG.relabel(gt_remaps)
        self.LP.relabel(pred_remaps)

        self.pred_inds_all_matched  = pred_inds_121+pred_inds_m21
        self.pred_labs_all_matched  = pred_labs_121+pred_labs_m2m
        self.pred_labs_unmatched = set(self.LP.labels).difference(set(self.pred_labs_all_matched))

        self.gt_inds_all_matched = gt_inds_121 +gt_inds_m21
        self.gt_labs_all_matched = self.gt_labs_121+self.gt_labs_m2m
        self.fks = fks_121+fks_m2m
        self.gt_labs_unmatched = set(self.LG.labels).difference(set(self.gt_labs_all_matched))

        prox_labels= list(zip(self.gt_labs_m2m,pred_labs_m2m))
        dsc_jac_multi = self._dsc_multilabel(prox_labels)
        self.dsc_multi = [a[0] for a in dsc_jac_multi]


# %%
if __name__ == "__main__":

    dsc = np.load("testfiles/dsc_test.npy")
    # LG = pd.read_csv("testfiles/LG.csv")
    # LP = pd.read_csv("testfiles/LP.csv")


    preds_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc"
    )
    gt_fldr = Path("/s/xnat_shadow/crc/lms_manual_final")
    gt_fns = list(gt_fldr.glob("*"))
    gt_fns = [fn for fn in gt_fns if is_sitk_file(fn)]

    case_subid = "CRC211"
    gt_fn = find_file(case_subid, gt_fns)
    pred_fn = find_file(case_subid, preds_fldr)
# %%
    # gt_fn = "testfiles/gt.nrrd"
    # pred_fn = "testfiles/pred.nrrd"
    S = ScorerFiles2(gt_fn,pred_fn,case_id ="abc",dusting_threshold=0)
    # one_to_many = np.argwhere(f_labs_matched > 1)

    # one_to_one = np.argwhere(m_labs_matched == 1)
    df = S.process()
# %%
    debug=False
    S.dust()
    S.gt_radiomics(debug)
    S.compute_overlap_overall()
    if S.empty_lm == "neither":
        S.compute_overlap_perlesion()
# %%


    row_counts = np.count_nonzero(S.dsc, 1)
    col_counts = np.count_nonzero(S.dsc, 0)
    
    pred_inds_m21=np.argwhere(col_counts>1).flatten().tolist()
    pred_inds_12x = np.argwhere(col_counts== 1).flatten().tolist()
    gt_inds_12m = np.argwhere(row_counts>1).flatten().tolist()

# %%
    fk_gen = fk_generator(0)
    dsc_inds_121=[]
    S.dsc_single=[]
    fks_121 , pred_inds_121,gt_inds_121 = [],[], []
    for pred_ind in pred_inds_12x:
        # pred_ind = pred_inds_x21[ind]
        row_ind = np.argwhere(S.dsc[:,pred_ind]>0)
        if np.count_nonzero(S.dsc[row_ind,:])==1:
            ind_pair = {'gt_ind':row_ind.item(), 'pred_ind':pred_ind}
            pred_ind_121 = pred_ind
            gt_ind_121 =row_ind.item()
            gt_inds_121.append(row_ind.item())
            pred_inds_121.append(pred_ind)
            S.dsc_single.append(S.dsc[gt_ind_121,pred_ind_121])
            fks_121.append(next(fk_gen))

# %%
    # np.argwhere(S.dsc[2,:]>0)

# %%
    gt_inds_m21 = S.corresponding_gt_inds(pred_inds_m21)
    inds = np.tril_indices(len(gt_inds_m21),-1)
    keep_inds=[True]*len(gt_inds_m21)
    gt_supersets = []
    for x,y in zip(*inds):
        set1 = gt_inds_m21[x]
        set2  = gt_inds_m21[y]
        if len(set1.intersection(set2))>0:
                keep_inds[x]=False
                keep_inds[y]=False
                gt_supersets.append(set1.union(set2))
    S.gt_inds_m2m = list(il.compress(gt_inds_m21,keep_inds))  + gt_supersets


# %%
    pred_inds_m2m ,fks_m2m= [],[]
    # gt_inds = gt_inds_m2m[0]
    for gt_inds in S.gt_inds_m2m:
        gt_inds = list(gt_inds)
        pred_inds = set(np.argwhere(S.dsc[gt_inds,:])[:,1])
        pred_inds_m2m.append(pred_inds)
        fks_m2m.append(next(fk_gen))

# %%

    S.gt_labs_121 = inds_to_labels(gt_inds_121)
    pred_labs_121 = inds_to_labels(pred_inds_121)


    gt_remaps, S.gt_labs_m2m = S.dsc_gp_remapping(S.gt_inds_m2m)
    pred_remaps, pred_labs_m2m = S.dsc_gp_remapping(pred_inds_m2m)

    S.LG.relabel(gt_remaps)
    S.LP.relabel(pred_remaps)

    S.pred_inds_all_matched  = pred_inds_121+pred_inds_m21
    S.pred_labs_all_matched  = pred_labs_121+pred_labs_m2m
    S.pred_labs_unmatched = set(S.LP.labels).difference(set(S.pred_labs_all_matched))
# %%

    S.gt_inds_all_matched = gt_inds_121 +gt_inds_m21
    S.gt_labs_all_matched = S.gt_labs_121+S.gt_labs_m2m
    S.fks = fks_121+fks_m2m
    S.gt_labs_unmatched = set(S.LG.labels).difference(set(S.gt_labs_all_matched))

# %%
# %%

    prox_labels= list(zip(S.gt_labs_m2m,pred_labs_m2m))
    dsc_jac_multi = S._dsc_multilabel(prox_labels)
    S.dsc_multi = [a[0] for a in dsc_jac_multi]


    S.LG.nbrhoods2 = S.insert_fks(S.LG.nbrhoods, -1, S.fks,S.gt_inds_all_matched,S.gt_labs_all_matched,S.gt_labs_unmatched)
# %%

    df = S.LG.nbrhoods.copy()
    colnames = ['label', 'cent','length','volume','label_cc', 'fk']
    df_neo =pd.DataFrame(columns=colnames)
    df['fk'] = dummy_fk
    df['label_cc_relabelled']=df['label_cc']
    for ind,fk in enumerate(fks):
        dsc_gp = dsc_gp_inds[ind]
        label = dsc_gp_labels[ind]
        if isinstance(dsc_gp,set):
            dsc_gp = list(dsc_gp)
            row= df.loc[dsc_gp]
        # if len(row)>1:
            label_dom = row['label'].max()
            cent = row['cent'].tolist()[0]
            length = row['length'].sum()
            volume = row['volume'].sum()
            df_dict = {'label':label_dom, 'cent':cent,'length':length, 'volume':volume,'label_cc':label,'fk':fk}
            # df_dict = pd.DataFrame(df_dict)
        else:
            row= df.loc[dsc_gp]
            df_dict = row[colnames].copy()
            df_dict['fk']= fk
            df_dict['label_cc']=label
        df_neo.loc[len(df_neo)]= df_dict

    for label_cc in dsc_labels_unmatched:
        row  = df.loc[df['label_cc']== label_cc ]
        df_dict = row[colnames].copy()
        # df_neo.loc[len(df_neo)]= df_dict
        df_neo = pd.concat([df_neo,df_dict],axis = 0,ignore_index=True)

# %%
