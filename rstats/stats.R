
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
setwd("/home/ub/code/mask_analysis")
# folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357"

folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/"
dfn <-file.path(folder,"results_all.csv")

# %%  functions

grep_names  <- function(x,names) names[grep(x,names)]
grep_df  <- partial(grep_names,names=names(df))

# %%

df <-read.csv(dfn,stringsAsFactors=TRUE)
relevant <- c("case_id","fn", "detected","original_shape_Maximum2DDiameterSlice","label")
partials <-c("dsc","jac")

relevant2 <- map(partials,grep_df) %>% unlist
df2 <- df[,c(relevant,relevant2)]
df2$detected <-  as.logical(df2$detected)
# %%
df_detected<- filter(df2,detected==TRUE)
df_small <- filter(df2,original_shape_Maximum2DDiameterSlice <10)

write.csv(df_small, file.path(folder,"df_sub_cm.csv"),row.names=FALSE)
write.csv(df2, file.path(folder,"df_relevant.csv"),row.names=FALSE)





# %%i#===========================PLOTS                                ================================
x <- seq(-pi,pi,0.1)
plot(x, cos(x))
# %%
