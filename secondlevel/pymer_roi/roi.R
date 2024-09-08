# Both hands
rm(list = ls())
# Import data
library(readxl)
data <- read_excel("/mnt/projects/PBCTI/ds-MRI-study/nilearn_stuff/scripts/task_scripts/second_level/pymer_roi/roi_lmm.xlsx")

library(lme4)


data$SubjectID = factor(data$SubjectID)
data$Run = factor(data$Run)
data$Reward = factor(data$Reward)
data$Gender = factor(data$Gender)
data$Grip = factor(data$Grip)

#first_run = data[data$Run==1,]
data_first_run = data[data$Run==1,]

#model <- lmer(Slope~Run+Reward*ROI+(1|SubjectID)+(1|SubjectID:Run),data=data)


model <- lmer(Slope ~ Age + Gender + EHI + Reward * ROI * Grip + (1 | SubjectID) + (1 | SubjectID:Grip), data = data_first_run)


#model <- lmer(ROI~Grip+Run+Reward*Slope+(1|SubjectID)+(1|SubjectID:Run),data=data)

model <- lmer(ROI~Grip*Reward*Slope+(1|SubjectID)+(1|SubjectID:Grip),data=data)

anova(model)
summary(model)



library(jtools)
library(interactions)
library(ggplot2)
library(emmeans)

cat_plot(model, pred = ROI, modx = Reward, geom = "line")

ggplot(data=data, aes(x=ROI,y=Slope, color=Reward)) + geom_line()

(mylist <- list(ROI=seq(-12,12,by=1),Reward=c("1","-1")))
emmip(model, Reward ~ROI, at=mylist,CIs=FALSE)


# Left hand
rm(list = ls())
# Import data
library(readxl)
data <- read_excel("/mnt/projects/PBCTI/ds-MRI-study/nilearn_stuff/scripts/task_scripts/second_level/pymer_roi/roi_lmm_lefthand.xlsx")

library(lme4)


data$SubjectID = factor(data$SubjectID)
data$Run = factor(data$Run)
data$Reward = factor(data$Reward)
data$Gender = factor


model <- lmer(Slope~Run+Age+EHI+Gender+Reward*ROI+(1|SubjectID)+(1|SubjectID:Run),data=data)

anova(model)
summary(model)


library(jtools)
library(interactions)
library(ggplot2)
library(emmeans)

cat_plot(model, pred = ROI, modx = Reward, geom = "line")

ggplot(data=data, aes(x=ROI,y=Slope, color=Reward)) + geom_line()

(mylist <- list(ROI=seq(-10,19,by=1),Reward=c("1","-1")))
emmip(model, Reward ~ROI, at=mylist,CIs=FALSE)




