# %%

library(purrr)
library(tidyr)
library(readxl)

library(dplyr)
setwd("/home/ub/code/label_analysis/label_analysis/rstats")

# folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357"

dfs_n<-file.path("/s/fran_storage/predictions/kits23/KITS23-SIRIG/results2/results2_thresh1mm_all.xlsx")
df <- read_excel(dfs_n)
# %%  functions
names(df)
# %%
#SECTION:-------------------- Case based DSC--------------------------------------------------------------------------------------
df_cases <- df %>% drop_na(gt_label_org) %>% distinct(case_id,gt_label_org,dsc,.keep_all=TRUE)

write.csv(df_cases,"cases.csv",row.names=FALSE)

# %%
