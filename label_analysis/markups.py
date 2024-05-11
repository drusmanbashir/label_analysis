# %%
from pathlib import Path
from fastcore.basics import store_attr

from label_analysis.helpers import get_labels, relabel, single_label
from label_analysis.geometry import LabelMapGeometry
import SimpleITK as sitk
from fran.utils.fileio import load_dict, load_json, maybe_makedirs, save_json
from fran.utils.string import find_file, info_from_filename, replace_extension, strip_extension
tmplt_folder = Path("/home/ub/code/label_analysis/label_analysis/markup_templates/")

class MarkupFromLabelmap():
    def __init__(self ,ignore_labels, dusting_threshold=3) -> None:
        self.load_templates()
        store_attr()

    def load_templates(self):
        self.main_dict= {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups":[]
        }
        markup_fn= tmplt_folder/("markup.json")
        self.markup_tmplt= load_json(markup_fn)
        self.empty_tmplt = load_json(tmplt_folder/("empty.json"))
        cp_fn = "label_analysis/markup_templates/controlpoint.json"
        self.cp_tmplt =load_json(cp_fn)
        self.key = load_json(self.template_json)

    @property
    def template_json(self):
        return tmplt_folder/("liver_AI.json")


    def process(self, lm_fn, outfldr=None, overwrite=False):
        lm_fn = Path(lm_fn)
        outfilename,outfldr = self.create_outfilename(outfldr,lm_fn)
        if overwrite==False and outfilename.exists():
            print("File {} already exists. Skipping..".format(outfilename))
            return 1
        maybe_makedirs(outfldr)
        case_props = info_from_filename(lm_fn.name)
        lm= sitk.ReadImage(lm_fn)
        lg = LabelMapGeometry(lm,ignore_labels=self.ignore_labels)
        lg.dust(self.dusting_threshold)
        if lg.is_empty():
                print("Writing EMPTY markup to",outfilename)
                save_json(self.empty_tmplt,outfilename)
        else:
            markups=[]
            for label in lg.labels_unique:
                schema_active= [dici for dici in self.key if dici['label']==label][0]
                str_rep = schema_active['str']
                color = schema_active['color']
                prefix = "_".join([case_props['proj_title'],case_props['case_id'],str_rep ])
                markup = self.create_label_markup(lg, label, prefix,color)
                markups.append(markup)
            print("Writing markup to",outfilename)
            dic_out = self.main_dict.copy()
            dic_out['markups']= markups
            save_json(dic_out, outfilename)


    def create_label_markup(self,lg, label:int, color:list, prefix:str, suffix=None):
        nbr_short= lg.nbrhoods.query('label=={}'.format(label))
        cps = []
        for ind in range(len(nbr_short)):
            id = str(ind+1)
            label = "-".join([prefix,id,suffix])
            orn = list(lg.lm_org.GetDirection())
            lesion = nbr_short.iloc[ind]
            pos =  list(lesion.cent)
            cp = self.create_controlpoint(id,label,pos,orn)
            cps.append(cp)
        markup = self.markup_tmplt.copy()
        markup['controlPoints']= cps
        markup['display']['color'] = color
        markup['display']['selectedColor'] = color
        markup['display']['activeColor'] = color
        return markup


    def create_controlpoint(self,id,label, position, orientation):
            cp = self.cp_tmplt.copy()
            cp['associatedNodeID'] = ''
            cp['id']= id
            cp['label']=  label
            cp['position'] = position
            cp['orientation']= orientation
            return cp


    def create_outfilename(self,outfldr,lm_fn ):

        if outfldr is None:
            outfldr = lm_fn.parent/("markups")
        fn_out = replace_extension(lm_fn.name,"json")
        fn_out = outfldr/fn_out
        return fn_out, outfldr

class MarkupDetectionOnly(MarkupFromLabelmap):
    '''
    This markup relabels every lesion to label 1. It does not discriminate
    '''

    def to_single_label(self,lm):
        labs = get_labels(lm)
        remapping = {x:1 for  x in labs}
        lm = relabel(lm,remapping)
        return lm

    @property
    def template_json(self):
        return tmplt_folder/("detection.json")
    def process(self,lm_fn, outfldr=None, overwrite=False):
        lm_fn = Path(lm_fn)
        outfilename,outfldr = self.create_outfilename(outfldr,lm_fn)
        if overwrite==False and outfilename.exists():
            print("File {} already exists. Skipping..".format(outfilename))
            return 1
        maybe_makedirs(outfldr)
        case_props = info_from_filename(lm_fn.name)

        lm= sitk.ReadImage(lm_fn)
        lm = self.to_single_label(lm)
        lg = LabelMapGeometry(lm,ignore_labels=self.ignore_labels)
        lg.dust(self.dusting_threshold)
        if lg.is_empty():
                print("Writing EMPTY markup to",outfilename)
                save_json(self.empty_tmplt,outfilename)
        else:
            markups=[]
            for label in lg.labels_unique:
                schema_active= [dici for dici in self.key if dici['label']==label][0]
                str_rep = schema_active['str']
                color = schema_active['color']
                prefix = "_".join([case_props['proj_title'],case_props['case_id'],str_rep ])
                markup = self.create_label_markup(lg, label, color,prefix,"AI")
                markups.append(markup)
            print("Writing markup to",outfilename)
            dic_out = self.main_dict.copy()
            dic_out['markups']= markups
            save_json(dic_out, outfilename)

   


# %%
if __name__ == "__main__":

    M = MarkupDetectionOnly(ignore_labels=[],dusting_threshold=0)
    # preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/")
    preds_fldr = Path("/s/fran_storage/predictions/lidc2/LITS-913_fixed_mc/")
    lm_fns = list(preds_fldr.glob("*.nii.gz"))
    lm_fns = ["/s/xnat_shadow/crc/lms_manual_final/crc_CRC138_20180812_Abdomen3p0I30f3.nii.gz"]
    # lm_fn = find_file("CRC084", lm_fns)
    # lm_fn = "/s/fran_storage/predictions/lidc2/LITS-913/lung_038.nii.gz"
    for lm_fn in lm_fns:
        M.process(lm_fn,overwrite=True)
# %%
    for lm_fn in lm_fns:
        M.process(lm_fn,overwrite=True)
    lm = sitk.ReadImage(lm_fn)
    labs = get_labels(lm)
    remapping = {x:1 for  x in labs}
    lm = relabel(lm,remapping)

    lg = LabelMapGeometry(lm,ignore_labels =[])
    lm_out = lm_fn.str_replace("mod","mod_cc")

    sitk.WriteImage(lg.lm_cc,lm_out)
# %%

    cp_fn = "label_analysis/markup_templates/controlpoint.json"
    cp_tmplt =load_json(cp_fn)
    tmplt_files = list(tmplt_folder.glob("*json"))
    schema = load_json(tmplt_folder/("schema_liverAI.json"))

# %%
    label =2
    schema_active= [dici for dici in M.key if dici['label']==label][0]
    str_rep = schema_active['str']
    color = schema_active['color']
    prefix = "_".join([case_props['proj_title'],case_props['case_id'],str_rep ])
    markup = M.create_label_markup(lg, label, color,prefix)

# %%
