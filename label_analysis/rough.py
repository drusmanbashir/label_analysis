from label_analysis.geometry_itk import *
import itk

# %%
#SECTION:-------------------- SETUp--------------------------------------------------------------------------------------
if __name__ == "__main__":

    lm2_fn = "/home/ub/Documents/nodes_90_411Ta_CAP1p5SoftTissue.nii.gz_2-Segment_1-label.nrrd"
    lm_fn = ("/s/fran_storage/predictions/lidc2/LITS-911/lung_020.nii.gz")
    lm_fn= "/s/fran_storage/datasets/raw_data/lidc/lms/lidc_0030.nii.gz";

    lm_fn= "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz"
    node_fn = "/s/xnat_shadow/nodes/lms/nodes_91_20201124_CAP5p0SoftTissue.nii.gz"
# %%
    L = LabelMapGeometryITK(node_fn)
    L.nbrhoods
    rows_rem = L.nbrhoods["flatness"]==0 
    L.remove_rows(rows_rem)
# %%
# %%
#SECTION:-------------------- ImageRegion--------------------------------------------------------------------------------------
    li = itk.imread(lm_fn, itk.UC)
    reg = li.GetLargestPossibleRegion()
    reg.GetNumberOfPixels()
# %%
    LIT = type(li)
    LO  = itk.LabelObject[itk.UL, 3]
    LMT = itk.LabelMap[itk.itkStatisticsLabelObjectPython.itkStatisticsLabelObjectUL3]
    LILM = itk.LabelImageToLabelMapFilter[itk.Image[itk.UC,3], LMT]
    CC = itk.ConnectedComponentImageFilter[itk.Image[itk.UC,3], itk.Image[itk.UL,3]]
    ToMap = LILM.New()
    ToMap.SetInput(li)
    ToMap.Update()
    ll = ToMap.GetOutput()
# %%
    L.nbrhoods= pd.DataFrame(rows)
    L.nbrhoods["bbox"] = L.nbrhoods["bbox"].apply(region_to_flat)

# %%
    dusting_threshold = 1
    removal = L.nbrhoods["feret"]<dusting_threshold
    nbr_tmp = L.nbrhoods[removal].copy()

    row = nbr_tmp.iloc[0]
    label_org = row["label_org"]
    lobj_ind = label_org-1
    label_cc = int(row["label_cc"])
    lmap= L.unique_lms[lobj_ind]['lmap']
    lmap.RemoveLabel(label_cc)
    dici = {"lmap":lmap,   "label_org":label_org,  "n_islands": lmap.GetNumberOfLabelObjects()}
    L.unique_lms[lobj_ind] = dici
# %%
    lm = sitk.ReadImage(lm_fn_out)


# %%

    lms = [lm["lmap"] for lm in L.unique_lms]
    merger = itk.MergeLabelMapFilter.New()
    M= itk.MergeLabelMapFilter[type(lms[0])]
    merger = M.New()
    merger.SetMethod(1)
    merger.SetInput(lms[0], *lms[1:])
    output = merger.GetOutput()
    out = itk.LabelMapToLabelImageFilter(output)
# %%
    A = itk.AddImageFilter[itk.Image[itk.UC,3]].New()
# %%

    itk.imwrite(A,lm_fn_out)
# %%
    im = sitk.ReadImage(lm_fn_out)
    get_labels(im)
    L1 = LabelMapGeometryITK(lm_fn_out)

# %%
#SECTION:-------------------- Relabel--------------------------------------------------------------------------------------
# %%

    fn_out = "/home/ub/tmp.nii.gz"
# %%
    df2 = pd.DataFrame(rows)
    print(df2)

    LSB = itk.LabelImageToShapeLabelMapFilter(L.lm_org)
    CC = itk.ConnectedComponentImageFilter(L.lm_org)
    lm2 = itk.cast_image_filter(CC,ttype=(type(CC), itk.Image[itk.UC,3]))
    LS = itk.LabelImageToShapeLabelMapFilter(lm2)
    oj1 = LS.GetLabelObject(2)

    print(len(oj1))
        key={}
        L.labels_org = get_labels(L.li_sitk)
        if len(L.labels_org) > 0:
            for lab in L.labels_org:
                lm = create_binary_image(L.li_org, lab, lab)
                # LO = itk.ShapeLabelObject[itk.UL, 3]
                LM = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,3]]
                Bin2S =  itk.BinaryImageToShapeLabelMapFilter[itk.Image[itk.UC,3], LM]
                f = Bin2S.New(Input=lm)
                f.SetComputeFeretDiameter(True)
                f.SetInputForegroundValue(1)
                # f.SetComputeOrientedBoundingBox(True)
                f.Update()
                lmap = f.GetOutput()
                k = {l:lab for l in lmap.GetLabels()}
                key.update(k)
                dici = {"lmap":lmap,   "label_org":lab,  "n_islands": lmap.GetNumberOfLabelObjects()}
                L.unique_lms.append(dici)
                for i in tqdm(range(lmap.GetNumberOfLabelObjects())):
                    obj = lmap.GetNthLabelObject(i)
                    rows.append({
                        "label_org": lab,
                        "label_cc": int(obj.GetLabel()),
                        "cent": tuple(obj.GetCentroid()),
                        "bbox": obj.GetBoundingBox(),
                        "flatness": float(obj.GetFlatness()),
                        "feret": float(obj.GetFeretDiameter()),
                        "volume_mm3": float(obj.GetPhysicalSize()),
                    })
            L.key = key


# %%
# %%
#SECTION:-------------------- BLOBS with unique labels as per original--------------------------------------------------------------------------------------
    oj1 = LSB.GetLabelObject(2)
    CC = itk.ConnectedComponentImageFilter(oj1)


# %%
    B2 = itk.LabelMapToLabelImageFilter(LS)
    itk.imwrite(B2,"/home/ub/tmp.nii.gz")

    CC.Update()

    BI = itk.BinaryImageToLabelMapFilter(L.lm_org)
    B2 = itk.LabelMapToLabelImageFilter(BI)


 %%
    RL = itk.RelabelComponentImageFilter(lm)
    RL.Update()
# %%
    lm_out = itk.cast_image_filter(CC,ttype=(type(CC), itk.Image[itk.UC,3]))
    itk.imwrite(RL,fn_out)
    itk.imwrite(lm_out, fn_out)

    writer= itk.ImageFileWriter[itk.Image[itk.UC,3]].New()
    writer.SetFileName(fn_out)
    writer.SetInput(CC)
    arr.max()

#SECTION:-------------------- ITK ROUTINE--------------------------------------------------------------------------------------

    lm2 = itk.imread(lm_fn, itk.UC)

    lm2.GetLabels()


    # L = LabelMapGeometryITK(lm2)
    L.lm.GetLabels()

    # PixelT = itk.template(lm)[1][0]
    # LO = itk.ShapeLabelObject[itk.UL, 3]
    LM = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,3]]

    Bin2S =  itk.BinaryImageToShapeLabelMapFilter[itk.Image[itk.UC,3], LM]

    f = Bin2S.New(Input=L.lm)
    f.SetComputeFeretDiameter(True)
    f.SetInputForegroundValue(1)
    # f.SetComputeOrientedBoundingBox(True)
    f.Update()
# %%
    def mini(lm, num_turns):
        start = time()
        for i in range(num_turns):
            print(i)
            # PixelT = itk.template(lm)[1][0]
            # LO = itk.ShapeLabelObject[itk.UL, 3]
            LM = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,3]]
            Bin2S =  itk.BinaryImageToShapeLabelMapFilter[itk.Image[itk.UC,3], LM]

            f = Bin2S.New(Input=lm)
            f.SetComputeFeretDiameter(True)
            f.SetInputForegroundValue(1)
            # f.SetComputeOrientedBoundingBox(True)
            f.Update()





            lmap=f.GetOutput()
        end = time()
        end-start
        print("Time taken: {}".format(end-start))
        return lmap

    from fran.utils.itk_sitk import (ConvertItkImageToSimpleItkImage,
                                     ConvertSimpleItkImageToItkImage)
    M = np.array(lm2.GetDirection()) if hasattr(lm2, "GetDirection") else np.eye(dim)
    lm3 = ConvertItkImageToSimpleItkImage(lm2,sitk.sitkUInt8, tuple(M.flatten()))
    lm4 = ConvertSimpleItkImageToItkImage(lm,itk.UC)
    lmap = mini(lm4,3)
    lf.GetLabels()
    binn = itk.LabelMapToBinaryImageFilter(lmap)
    itk.imwrite(binn,"bin.nii.gz")





# %%

    fn = "bin.nii.gz"

    lm2 = itk.imread(fn,itk.UC)

# %%

    lf = itk.LabelImageToLabelMapFilter(lm2)
    lf.Update()
    lf.GetLabels()
# %%

# %%
    rows = []
    for i in range(lmap.GetNumberOfLabelObjects()):
        obj = lmap.GetNthLabelObject(i)
        rows.append({
            "label": int(obj.GetLabel()),
            "feret": float(obj.GetFeretDiameter()),
            "volume_mm3": float(obj.GetPhysicalSize()),
            "centroid_phys": tuple(obj.GetCentroid()),
        })
    df2 = pd.DataFrame(rows)

    print(df2)
# %%
    LGF = itk.LabelGeometryImageFilter[itk.Image[itk.UC,3]]
    itk.LabelImageToShapeLabelMapFilter
    
# %%
    # bina = itk.BinaryImageToLabelMapFilter(itk.Image[itk.UC,3], itk.LabelMap[itk.StatisticsLabelObject[itkL,
    bina.SetInput(lm2)
    bina.Update()
    lm2 = bina.GetOutput()
    itk.template(l2)
# %%
# %%

# %%




    LG = LabelMapGeometry(lm2)
    lm_cc = to_cc(lm2)

    fn1 = "gt_lm_cc_unaltered.nii.gz"
    sitk.WriteImage(lm_cc, fn1)
# %%

# %%
    lm_f2 = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/crc_CRC211_20170724_AbdoPelvis1p5.nii.gz"
    lm2 = sitk.ReadImage(lm_f2)
    LP = LabelMapGeometry(lm2,[1])
    LP.nbrhoods
    lm_cc2 = to_cc(lm2)

    fn2 = "pred_lm_cc_unaltered.nii.gz"
    sitk.WriteImage(lm_cc2, fn2)
# %%

    print(get_labels(lm2))
    lm2 = relabel(lm2,{1:3})
    sitk.WriteImage(lm2, lm_f)
# %%
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
