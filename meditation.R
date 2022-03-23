model_data <- read.csv('meditation_data.csv', header = TRUE)
head(model_data)

library("lavaan")
isi_mean = model_data[,14]
sds = model_data[,4]

bmi = model_data[,2]
gender = model_data[,3]
anxiety = model_data[,5]
meq = model_data[,6]
first = model_data[,8]
isi_base = model_data[,9]
fnc_mean = model_data[,15]

fullData <- data.frame(X = sds, Y = isi_mean, M1 = fnc_mean,
                       E1 = bmi, E2 = gender, E3 = anxiety, E4 = meq, 
                       E5 = first, E6 = isi_base)
# set model
# contrast test whether differ significantly
# test direct effect to see whether is fully mediation

Mediation <- '
Y ~ b1* M1 + c*X + eb1*E1 + eb2*E2 + eb3*E3 + eb4*E4 + eb5*E5 + eb6*E6
M1~ a1 * X
indirect1 := a1 * b1
dirtect := c
total := c + a1 * b1
'

fit <- sem(model = Mediation, data = fullData, se = 'bootstrap', bootstrap = 10000)
summary(fit)

# see confidence interval
parameterEstimates(fit, boot.ci.type="bca.simple")