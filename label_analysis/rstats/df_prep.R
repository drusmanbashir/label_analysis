require(colorout)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
setwd("/home/ub/code/mask_analysis/rstats")

# folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357"

folder <-"/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/"
dfn <-file.path(folder,"results_all_small.csv")
df2n <-"../results/results.csv"

# %%  functions

grep_names  <- function(x,names) names[grep(x,names)]
max_dsc <-function(row){
  vals <-row[dsc_labels]
  maxval<-max(vals[!is.na(vals)])
  return(as.numeric(maxval))
}
# %%

df1 <-read.csv(dfn,stringsAsFactors=TRUE)
df2<-read.csv(df2n,stringsAsFactors=TRUE)
df2$fp<-str_replace(as.character(df2$fp),"\\[\\]","")

# %%
relevant <- c("case_id","fn", "detected","original_shape_Maximum2DDiameterSlice","label","fp")
partials <-c("dsc")
final_cols <- c('case_id' , 'original_shape_Maximum2DDiameterSlice' , 'dsc_max','detected','fp')

#----------------------- Merge labels into 1 -----------------------------------




# %%
dfs<-list(df1,df2)
dfo <-lapply(dfs, function(df){

  df$detected <-  as.logical(df$detected)
  grep_df  <- partial(grep_names,names=names(df))
  dsc_labels<- map(partials,grep_df) %>% unlist
  dsc_labels<- dsc_labels[dsc_labels!="dsc_overall"]
  df$dsc_max<-apply(df,1,max_dsc)
  df<-df[,final_cols]
  return(df)

})

df<-rbind(dfo[[1]],dfo[[2]])
 
# %% ------------------------------------ 




df <- df %>%mutate(sz_bands = cut(original_shape_Maximum2DDiameterSlice,breaks = c(0,10,20,Inf)))

df<-df[c('case_id' , 'sz_bands','original_shape_Maximum2DDiameterSlice' , 'dsc_max','detected','fp')]

write.csv(df,file.path(folder,"results_rsna.csv"))
save(df,file = "df.Rda")
# %%
# %%
df_small<-df %>% filter(sz_bands=="(0,10]")
small<-df %>% filter(dsc_max>0.1 & sz_bands=="(0,10]") %>% nrow() /(summ[1,3])
med<-df %>% filter(dsc_max>0.1 & sz_bands=="(10,20]") %>% nrow() /summ[2,3]
large<-df %>% filter(dsc_max>0.1 & sz_bands=="(20,Inf]") %>% nrow() /summ[3,3]
print(c(small,med,large))





# %%
df2[df2$dsc_bands[1]]
save.image()
# %%




df_detected<- filter(df2,detected==TRUE)
df_small <- filter(df2,original_shape_Maximum2DDiameterSlice <10)

write.csv(df_small, file.path(folder,"df_sub_cm.csv"),row.names=FALSE)
write.csv(df, file.path(folder,"df_relevant.csv"),row.names=FALSE)





# %%i#===========================PLOTS                                ================================
# %%





# %%--------------------------------------------- ROUGH ----------------------------------------------------------
save(file="r.RData")
set.seed(549298)                       # Create example data
data <- data.frame(x = rnorm(50, 1, 3),
                   group = LETTERS[1:5])
# %%
