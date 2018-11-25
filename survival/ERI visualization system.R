setwd('C:/Users/longtan/Desktop/00-Code Project/05-ERI')
require(ggplot2)
library("ggthemes")
library("reshape2")
library("plyr")

ERI_sys<-read.csv("ERI system.csv",sep=';',header=T)
ERI_sys<-ERI_sys[1:73,]
ERI_sys$Engine.production.week<-factor(ERI_sys$Engine.production.week,levels=ERI_sys$Engine.production.week)
ggplot(ERI_sys,aes(x=Engine.production.week,y=X50._pph,group=1))+
geom_ribbon(aes(ymin=X5._pph,ymax=X95._pph),fill='grey70',alpha=0.5)+
geom_line(size=1,color='red')+ylim(0,75)+
scale_x_discrete(breaks=ERI_sys[,'Engine.production.week'][seq(0,nrow(ERI_sys),5)])+
theme_wsj(color="brown")