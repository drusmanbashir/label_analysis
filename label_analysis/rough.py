# %%
import re
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite as bp
import pandas as pd
import numpy as np
from label_analysis.helpers import get_labels, inds_to_labels, relabel
import itertools as il
import networkx as nx
from fran.utils.string import find_file
from scipy.sparse import csr, csr_matrix 
import SimpleITK as sitk
# %%

def fk_generator(start=0):
    key = start
    while True:
        yield key
        key+=1

# %%
# %%
if __name__ == "__main__":

    lm_f = "/s/xnat_shadow/crc/lms/crc_CRC306_20161210_Abdomen3p0I30f3.nii.gz"
    lm = sitk.ReadImage(lm_f)
    lm = relabel(lm,{2:3})
    sitk.WriteImage(lm, lm_f)
    get_labels(lm)
    dsc = np.load("testfiles/dsc_CRC171.npy")
    fn2 = "/s/xnat_shadow/nodes_thick/images/nodes_5_20190529_Abdomen_thick.nii.gz"
    fn1 = "/home/ub/nodes_5_20190529_Abdomen_thick.nii.gz_1.nii"
    im = sitk.ReadImage(fn1)
    im.GetOrigin()
    im2 = sitk.ReadImage(fn2)

    r = dsc.shape[0]
    s = dsc.shape[1]
    gt_labs = ['gt_lab_'+str(x+1) for x in range(r)]
    pred_labs = ['pred_lab_'+str(x+1) for x in range(s)]

    G = nx.Graph()
    G.add_nodes_from(gt_labs,bpartite=0)
    G.add_nodes_from(pred_labs,bpartite=1)
# %%
    edge_sets=[]
    for row in range(r):
        gt_lab = "gt_lab_"+str(row+1)
        edges =np.argwhere(dsc[row,:]).flatten().tolist()
        pred_labs = ["pred_lab_"+str(ind+1) for ind in edges]
        gt_lab_2_preds=[(gt_lab, pred_lab) for pred_lab in pred_labs]
        edge_sets.extend(gt_lab_2_preds)
    

    # G.add_edges_from()
# %%
    G.add_edges_from(edge_sets)
    con= nx.connected_components(G)
    ccs= list(con)
    

    fk_gen = fk_generator(start=1)
    gt_remaps={}
    pred_remaps = {}
    fks=[]

    for cc in ccs:
        fk = next(fk_gen)
        fks.append(fk)
        while(cc):
            label = cc.pop()
            indx = re.findall(r'\d+',label )[0]
            indx = int(indx)
            if 'gt' in label:
                gt_remaps.update({indx:fk})
            else:
                pred_remaps.update({indx:fk})
# %%
    nx.draw(G,with_labels=True)
    plt.show()

# %%
    # LG = pd.read_csv("testfiles/LG.csv")
    # LP = pd.read_csv("testfiles/LP.csv")

# %%
# Initialise graph

# Add nodes with the node attribute "bipartite"
top_nodes = [1, 2, 3]
bottom_nodes = ["A", "B", "C"]
B = nx.Graph()
B.add_nodes_from(top_nodes, bipartite=0)
B.add_nodes_from(bottom_nodes, bipartite=1)
a= nx.connected_components(B)
# Add edges only between nodes of opposite node sets
B.add_edges_from([(1, "B"),  (2, "A"), (2, "C"), (3, "A"), (3, "C")])
nx.draw(B,with_labels=True)
# %%
pos = dict()
X,Y = bp.sets(B)
pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
nx.draw(B, pos=pos,with_labels=True)
# %%
