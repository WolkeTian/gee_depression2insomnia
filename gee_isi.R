modeldata = read.csv("insomnia_fnc_gee.csv", header = TRUE)
head(modeldata)

library(geepack)
gee_isi = geeglm(formula = isi ~ bmi + gender + sds +
                   anxiety + meq + first +isi_base,
                  family = gaussian, data = modeldata, 
                 id = subid, corstr = "exchangeable")

summary(gee_isi)

# compare qic
QIC(gee_isi)
