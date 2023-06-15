# %%
library(dplyr)
library(purrr)
library(tidyr)
# %%
setwd("/home/ub/code/mask_analysis/rstats")

# folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357"

folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/"
dfs_n<-file.path(folder,"df_sub_cm.csv")
# df_n<-file.path(folder,"df_relevant.csv")
dfa_n <-file.path(folder,"results_rsna.csv")
dfs<- read.csv(dfs_n)
dfn<-read.csv("../results/normal_cases_analysis.csv")

load ('df.Rda')

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

