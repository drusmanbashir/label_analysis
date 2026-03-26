# %%
library(dplyr)
library(purrr)
library(tidyr)
library(readxl)
# %%
setwd("/home/ub/code/label_analysis/label_analysis/rstats")
library(devtools)

# install.packages("tidyverse")
# install.packages("rlang")
# install.packages("nvimcom")
library('ggplot2')
library('tidyverse')

# devtools::install_github("r-lib/rlang")
# %%
#SECTION:-------------------- section--------------------------------------------------------------------------------------
# folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357"

# %%
redundant  <- c("lm_filename","minor_axis", "least_axis","processing_error","error_type", "error_message", "label")
folder <-"/s/fran_storage/predictions/kits2/KITS2-bk/results"
dfs_n<-file.path(folder,"results_thresh1mm_all.xlsx")
df2_fn = "/r/datasets/preprocessed/test/fixed_spacing/spc_080_080_150_rsc5609df8a/dataset_stats/lesion_stats2.csv"

df_fn = "/r/datasets/preprocessed/test/fixed_spacing/spc_080_080_150_rsc5609df8a/dataset_stats/lesion_stats.csv"
df_fn = "/media/UB/datasets/kits23/dataset_stats/lesion_stats.csv"
# df_n<-file.path(folder,"df_relevant.csv")
df  <- read_csv(df_fn)
bad_rows = df$label_org==1
df2 = df[!bad_rows,]
df2 = df2[, -which(names(df2) %in% redundant)]

cid = "kits23_00020"

df2[df2$case_id==cid,]


df2.to_csv(df2_fn)
write.csv(df2,df2_fn)
# %%
df[df$'1'=="loss_dice_label3",]
df  %>% group_by(case_id) %>% summarise(md = median(2))
df2= df[df$case_id=="kits23_00000",]
df3 = df2[df2$label_org!=1,]
df3 = df2[df2$label_org!=1, -which(names(df) %in% redundant)]

# %%
#SECTION:-------------------- section--------------------------------------------------------------------------------------
# COUNT FP 
# %%


get_num_fp <-function(fp_col){
  st <-fp_col[[1]]
  if (st=="[]"){ll = 0}else{
  l <-strsplit(st,",")
  ll <-unlist(length(l[[1]]))
  }
  return (ll)
}
# %%
df$fp_num<-map(df$fp,get_num_fp)
df$fp_num<-unlist(df$fp_num)

fp_counts_df <-df[,c("case_id","fp_num")]
fp_counts_df <-fp_counts_df[!duplicated(fp_counts_df$case_id), ]
# %%
dfn$fp_num<- map(dfn$lengths,get_num_fp)
dfn<-dfn%>% filter(selected!='no')
df%>% distinct(case_id)

dfn %>% filter(selected=="fp")

# %%
table(dfn$selected)
# %% [markdown]
## 
summ<-df %>%group_by(sz_bands) %>%summarise(
  median = median(dsc_max),
  nrows = length(sz_bands)
)

# %%
print(summ)

summary(df[c("sz_bands", "detected")])
tapply(df$sz_bands,df$detected,summary)

# %%
137/190
# %%

# %%

