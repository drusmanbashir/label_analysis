# %%
from pathlib import Path
from fastcore.basics import store_attr

from label_analysis.helpers import get_labels, relabel, single_label, to_cc
from label_analysis.geometry import LabelMapGeometry
import SimpleITK as sitk
from utilz.fileio import load_dict, load_json, maybe_makedirs, save_json
from utilz.imageviewers import view_sitk
from utilz.string import find_file, info_from_filename, replace_extension, strip_extension
tmplt_folder = Path("/home/ub/code/label_analysis/label_analysis/markup_templates/")


class MarkupFromLabelmap():
    '''
    
    Each lesion is assigned a markup controlpoint. The controlpoint properties are based on the lesion label. This is in turn dicated by  ...
    template used

    '''
    
    def __init__(self ,ignore_labels, dusting_threshold=3,template='auto',color=None, shape="Sphere3D") -> None:
        assert template in ['auto','liver'], "Pick a template; either 'auto' or 'liver' "
        if color:
            assert color in self.color_LUT.keys(), "Color has to be one of {}".format(self.color_LUT.keys())
        store_attr()
        self.load_templates()


    @property
    def color_LUT(self):
        return{
            'red':[1.0,0.0,0.0],
            'green':[0.0,1.0,0.0],
            'blue':[0.0,0.0,1.0],
            'yellow':[1,1,0],
        }


    def load_templates(self):
        self.main_dict= {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups":[]
        }
        markup_fn= tmplt_folder/("markup.json")
        self.markup_tmplt= load_json(markup_fn)
        self.empty_tmplt = load_json(tmplt_folder/("empty.json"))
        cp_fn = tmplt_folder/("controlpoint.json")
        self.cp_tmplt =load_json(cp_fn)
        self.key = load_json(self.template_json)

    @property
    def template_json(self):
        if self.template =='liver':
            return tmplt_folder/("liver_AI.json")
        elif self.template =='auto':
            return tmplt_folder/("auto.json")


    def process(self, lm) -> dict:
        lg = LabelMapGeometry(lm,ignore_labels=self.ignore_labels)
        lg.dust(self.dusting_threshold)
        if lg.is_empty():
                return self.empty_tmplt
        else:
            markups=[]
            for label in lg.labels_unique:
                schema_active= [dici for dici in self.key if dici['label']==label][0]
                str_rep = schema_active['str']
                if self.color:
                    color = self.color_LUT[self.color]
                else:
                    color = schema_active['color']
                prefix = "_".join(['projtitle','caseid',str_rep ])
                markup = self.create_label_markup(lg, label, color,prefix)
                markups.append(markup)

            dic_out = self.main_dict.copy()
            dic_out['markups']= markups
            return dic_out

    def create_label_markup(self,lg, label:int, color:list, prefix:str, suffix=''):
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

        if self.shape is not None:
            markup["display"]["glyphType"] = self.shape
        return markup


    def create_controlpoint(self,id,label, position, orientation):
            cp = self.cp_tmplt.copy()
            cp['associatedNodeID'] = ''
            cp['id']= id
            cp['label']=  label
            cp['position'] = position
            cp['orientation']= orientation
            return cp


class MarkupMultipleFiles(MarkupFromLabelmap):
    def process(self,lm_fns, out_filename):
        self.total= 0
        ind=0
        # mups_tmp=[]
        for fn in lm_fns:
            lm = sitk.ReadImage(str(fn))
            mup = super().process(lm)
            if ind==0:
                mups_final = mup
                cps  =mup['markups'][0]['controlPoints']
                self.total+=len(cps)
                # mups_tmp.append(mup)
            else:
                cps = mup['markups'][0]['controlPoints']
                self.total+=len(cps)
                print("Number of contol points in {0}: {1}".format(fn, len(cps)))
                # mups_final = mups_tmp[0]
                mups_final['markups'][0]['controlPoints'].extend(cps)
            ind+=1

        save_json(mups_final,out_filename)

    def __len__(self): return self.total


class MarkupFromLabelFile(MarkupFromLabelmap):

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


    def create_outfilename(self,outfldr,lm_fn ):

        if outfldr is None:
            outfldr = lm_fn.parent/("markups")
        else:
            outfldr = Path(outfldr)
        fn_out = replace_extension(lm_fn.name,"json")
        fn_out = outfldr/fn_out
        return fn_out, outfldr

class MarkupDetectionOnly(MarkupFromLabelFile):
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

# %%
#SECTION:--------------------  setup--------------------------------------------------------------------------------------

    # preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/")
    preds_fldr = Path("/s/fran_storage/predictions/lidc2/LITS-913_fixed_mc/")
    gt_fldr = Path("/s/xnat_shadow/crc/lms_manual_final/")
    nodes_fldr = Path("/s/xnat_shadow/nodesthick/lms")
    outfldr = nodes_fldr.parent/("markups")
    lm_fns = list(nodes_fldr.glob("*.nii.gz"))

    lm_fns = ["/s/xnat_shadow/nodes/lms/nodes_70_20210804_ChestAbdomenPelviswithIVC1p00Hr40S3.nii.gz"]
    # lm_fn = "/s/fran_storage/predictions/lidc2/LITS-913/lung_038.nii.gz"
    maybe_makedirs([outfldr])

# %%
# %%
#SECTION:-------------------- Add markups--------------------------------------------------------------------------------------

    M = MarkupDetectionOnly(ignore_labels=[],dusting_threshold=0 )
# %%
    for lm_fn in lm_fns:
        M.process(lm_fn,overwrite=True,outfldr=outfldr)
# %%
#SECTION:-------------------- all labels remapped to LABEL 1 
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
#SECTION:-------------------- ROUGH--------------------------------------------------------------------------------------

    lm_fn = lm_fns[0]
    lm = sitk.ReadImage(lm_fn)
    lm_cc = to_cc(lm)
# %%
    di = sitk.SignedMaurerDistanceMap(lm!=0,insideIsPositive=False,squaredDistance=False,useImageSpacing=False)
    radius = .3
    seeds = sitk.ConnectedComponent(di<radius)
    ws = sitk.MorphologicalWatershedFromMarkers(di, seeds, markWatershedLine=True)
    ws2 = sitk.Mask( ws, sitk.Cast(lm, ws.GetPixelID()))
# %%
    view_sitk(ws2,ws,'mm')
# %%

    radius2 = 30
    seeds2 = sitk.ConnectedComponent(di<radius2)
    ws3 = sitk.MorphologicalWatershedFromMarkers(di, seeds2, markWatershedLine=True)
    ws4 = sitk.Mask( ws3, sitk.Cast(lm, ws.GetPixelID()))

    view_sitk(ws3,ws,'mm')
# %%
# %%
