# %%
import pandas as pd
import ipdb
from pyxnat.core.resources import shutil
from fran.utils.imageviewers import view_sitk
import ipdb
tr = ipdb.set_trace

import itertools as il
from pathlib import Path
from fran.utils.fileio import load_dict
from fran.utils.helpers import get_pbar
import SimpleITK as sitk

from fran.utils.string import find_file, strip_extension

class  Jack():
    def __init__(self,jack):
        self.jack = jack

    def out(self):
        print (self.jack)



def df_to_dest(cid_df, dest_fldr,all_files):
        '''
        outputs a df with columns 'fname_in','fname_out',  which may be used to move files
        cid_df only has case_ids column. moves every  matching fn in all_files to dest_fldr
        dest_folder will have subfolder for images and masks
        
        '''
        
        def _inner(fname):
                        dad_img = fname.parent.name =='images'
                        name = fname.name
                        if dad_img==True:
                            outfname = dest_im/name
                        else:
                            outfname = dest_masks/name
                        sublist = [str(fname), str(outfname)]
                        return sublist

        dest_im = dest_fldr/("images")
        dest_masks = dest_fldr/("masks")
        listi =[]
        for ind in range(len(cid_df)):
            cid = cid_df.iloc[ind]
            print(cid)
            try:
                fn = find_file(cid,all_files)
                if isinstance(fn,list):
                    for fname in fn:
                        listi.append(_inner(fname))
                else:
                    listi.append(_inner(fn))
            except ValueError:
                pass
        df_out = pd.DataFrame(listi,columns=['fname_in','fname_out'])
        return df_out

# %%
if __name__ == "__main__":


    im_srn = "/s/xnat_shadow/crc/srn/images"
    lm_srn = "/s/xnat_shadow/crc/srn/masks"
    fldrs = [Path(fl) for fl in [im_srn,lm_srn]]
# %%
    fls=[]
    for fldr in fldrs:
        files = list(fldr.glob("*"))
        fls.append(files)
# %%
    all_files = list(il.chain.from_iterable(fls))
    print(len(all_files))
# %%
    fn = "/s/xnat_shadow/crc/srn/srn_summary_positive_cases_only.csv"
    df = pd.read_csv(fn)
    df.columns
    df2 = df['case_id']
    pd.unique(df['labels'])
    wxh  = 'Whipps Cross University Hospital'

    wx_c= df[df['hospital']==wxh].case_id.dropna()
    other_c = df[df['hospital']!=wxh].case_id.dropna()
    cid = "crc_CRC319"
    cid  in wx_c

# %%
    dest_fldr = Path("/s/xnat_shadow/crc/srn/cases_with_findings/")
    dest_fldr2 = Path("/s/xnat_shadow/crc/srn")
    df_out1 = df_to_dest(df2, dest_fldr,all_files)
    df_out2 = df_to_dest(other_c, dest_fldr2,all_files)
# %%
    for row in df_out1.itertuples():
        if not Path(row.fname_out).exists():
            shutil.copy(row.fname_in,row.fname_out)

# %%
    wxh_im = list((dest_fldr/("images")).glob("*"))
    others = list((dest_fldr2/("masks")).glob("*"))

    n1 = [fn.name for fn in wxh_im]
    n2 = [fn.name for fn in others]
    set(n1).intersection(set(n2))
# %%


