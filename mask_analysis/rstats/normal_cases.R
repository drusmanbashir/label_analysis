require(colorout)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
setwd("/home/ub/code/mask_analysis/rstats")

# folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357"

folder <-"../results"
dfn <-file.path(folder,"normal_cases_analysis.csv")

# %%
df <-read.csv(dfn)
names(df)
df['selected']
df2<-df %>% filter (selected!='no')
