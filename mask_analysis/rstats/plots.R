library(dplyr)
library(ggplot2)
getwd()
load("df.Rda")
names(df)
# %%
ggplot(df,aes(x=sz_bands,y=dsc_max))  + geom_boxplot()
# %%
dfs<- df%>% filter(sz_bands=="(0,10]")
df%>% filter(sz_bands=="(10,20]" & dsc_max<0.3)
# %%


