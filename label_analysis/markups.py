# %%
from pathlib import Path
from fastcore.basics import store_attr

from label_analysis.overlap import LabelMapGeometry
import SimpleITK as sitk
from fran.utils.fileio import load_dict, load_json, maybe_makedirs, save_json
from fran.utils.string import find_file, info_from_filename, replace_extension, strip_extension
tmplt_folder = Path("/home/ub/code/label_analysis/label_analysis/markup_templates/")

class MarkupFromLabelmap():
    def __init__(self ) -> None:
        self.dusting_threshold = 3
        self.load_templates()

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
        self.key = load_json(tmplt_folder/("key_liverAI.json"))

    def process(self, lm_fn, outfldr=None, overwrite=False):
        lm_fn = Path(lm_fn)
        if outfldr is None:
            outfldr = lm_fn.parent/("markups")
        outfilename = self.create_outfilename(outfldr,lm_fn)
        if overwrite==False and outfilename.exists():
            print("File {} already exists. Skipping..".format(outfilename))
            return 1
        maybe_makedirs(outfldr)
        case_props = info_from_filename(lm_fn.name)
        lm= sitk.ReadImage(lm_fn)
        lg = LabelMapGeometry(lm)
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


    def create_label_markup(self,lg, label:int, prefix:str, color:list):
        nbr_short= lg.nbrhoods.query('label=={}'.format(label))
        cps = []
        for ind in range(len(nbr_short)):
            id = str(ind)
            lab = "-".join([prefix,str(ind)])
            orn = list(lg.lm_org.GetDirection())
            lesion = nbr_short.iloc[ind]
            pos =  list(lesion.cent)
            cp = self.create_controlpoint(id,lab,pos,orn)
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
        fn_out = replace_extension(lm_fn.name,"json")
        fn_out = outfldr/fn_out

        return fn_out

# %%
if __name__ == "__main__":

    M = MarkupFromLabelmap()
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/")
    lm_fns = list(preds_fldr.glob("*.nii.gz"))
    lm_fn = find_file(lm_fns, "CRC084")
    for lm_fn in lm_fns:
        M.process(lm_fn,overwrite=True)
    lm = sitk.ReadImage(lm_fn)
    lg = LabelMapGeometry(lm)
    lm_out = lm_fn.str_replace("mod","mod_cc")
    sitk.WriteImage(lg.lm_cc,lm_out)
# %%

    cp_fn = "label_analysis/markup_templates/controlpoint.json"
    cp_tmplt =load_json(cp_fn)
    tmplt_files = list(tmplt_folder.glob("*json"))
    schema = load_json(tmplt_folder/("schema_liverAI.json"))
# %%
