setwd('C:/Users/longtan/Desktop/00-Code Project/05-ERI')
require(ggplot2)
library("ggthemes")
library("reshape2")
library("plyr")

Da<-read.csv("Issue_list.csv",sep=';',header=T)
Da<-data.frame(Da)
Data<-Da[c("Complaints","X10.0","X22")]
Data<-Data[order(Data$X22,decreasing=T),]
Data<-Data[1:10,]
Data<-Data[order(Data$X22),]
Data$Complaints<-factor(Data$Complaints,levels=Data$Complaints)
p<-ggplot(Data)+
geom_segment(aes(Complaints,X10.0,xend=Complaints,yend=X22),linetype='solid',size=1,color="steelblue2",arrow=arrow(length = unit(0.02, "npc")))+
geom_bar(aes(x=Complaints,y=X10.0),stat = "identity",fill = "steelblue1",width=0.4)+
geom_point(aes(x=Complaints,y=X22),color="black",size=0.5)+coord_flip()+
scale_y_continuous("Failure Probability [%]",limits=c(0,0.1),labels=scales::percent)+theme_wsj(color="brown")+
theme(axis.line.y = element_line(colour = "black"),axis.text.y=element_text(size=12),axis.text.x=element_text(size=9))+labs(x="Top issues")