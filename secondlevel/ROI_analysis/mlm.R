rm(list = ls())

library(lme4)

data_path = '/mnt/projects/PBCTI/ds-MRI-study/nilearn_stuff/scripts/task_scripts/second_level/ROI_analysis/roi_outputs_gm.csv'

data= read_csv(data_path)

data$SubjectID = factor(data$SubjectID)
data$Run = factor(data$Run)
data$Reward = factor(data$Reward)
data$GripResponse = factor(data$GripResponse)


accumbensL_model <- lmer(AccumbensL~Reward*Slope*mdi+bisbas_bas+bisbas_bis+GripResponse+Run + sex + (1|SubjectID)+(1|SubjectID:Run), data=data)
anova(accumbensL_model)

accumbensR_model <- lmer(AccumbensR~Reward*Slope*mdi+bisbas_bas+bisbas_bis+GripResponse+Run + sex + (1|SubjectID)+(1|SubjectID:Run), data=data)
anova(accumbensR_model)

putamenL_model <- lmer(PutamenL~Reward*Slope*mdi+bisbas_bas+bisbas_bis+GripResponse+Run + sex + (1|SubjectID)+(1|SubjectID:Run), data=data)
anova(putamenL_model)

putamenR_model <- lmer(PutamenR~Reward*Slope*mdi+bisbas_bas+bisbas_bis+GripResponse+Run + sex + (1|SubjectID)+(1|SubjectID:Run), data=data)
anova(putamenR_model)

caudateL_model <- lmer(CaudateL~Reward*Slope*mdi+bisbas_bas+bisbas_bis+GripResponse+Run + sex + (1|SubjectID)+(1|SubjectID:Run), data=data)
anova(caudateL_model)

caudateR_model <- lmer(CaudateR~Reward*Slope*mdi+bisbas_bas+bisbas_bis+GripResponse+Run + sex + (1|SubjectID)+(1|SubjectID:Run), data=data)
anova(caudateR_model)




