# Script que carga un archivo de datos csv
# hace la regresión lineal con una variable independiente
# luego se hace un análisis estadístico para validar
# la significación de la correlación obtenida
# Linear regression for a two-variable dataset
# Then we perform a statistic analysis to verify
# the significance of the correlation obtained
# Dec 2021. Carlos Utrera
import pandas as pd
from sklearn.linear_model import LinearRegression
from math import sqrt
import scipy.stats as spstat
from scipy.stats import t
df = pd.read_csv('linear_normal.csv', delimiter=",")
X = df.values[:,0]
Y = df.values[:,1]
n = Y.size
fit = LinearRegression().fit(X.reshape(-1,1), Y)
slope=fit.coef_[0]
intercept=fit.intercept_
print("y = {0} + {1}x".format(fit.intercept_, fit.coef_[0]))
# Se hará una evaluación estadística del modelo obtenido
# calculo el coeficiente de correlación manualmente
numerator=X.size*sum(X*Y)-sum(X)*sum(Y)
denominator=(X.size*sum(X**2)-sum(X)**2)*(Y.size*sum(Y**2)-sum(Y)**2)
denominator=sqrt(denominator)
rCoef=numerator/denominator
rMat=df.corr(method='pearson')
rStat= spstat.pearsonr(X,Y)
# Calculo el test value
testValue=rCoef/sqrt( (1- rCoef*rCoef ) / (n - 2))
if testValue > 0:
    pValue=1-t(n-1).cdf(testValue)
else:
    pValue=t(n-1).cdf(testValue)
lowVal=t(n-1).ppf(0.025)
uppVal=t(n-1).ppf(0.975)
error=sqrt(sum((Y-(slope*X+intercept))**2)/(n-2))
print ("Coeficiente de pandas:",rMat.iat[1,0])
print("Coeficiente calculado from scratch:",rCoef)
print("Coeficiente por scipy:",rStat[0],"p-value:",rStat[1])
print("Test value {}",format(testValue))
print("Critical range: {} to {}".format(lowVal,uppVal))
if testValue > uppVal or testValue < lowVal:
    print("Hipótesis alternativa válida, correlación estadísticamente significativa")
else:
    print("Hipótesis nula válida, correlación estadísticamente NO significativa")
print("p-value: {}\nError:{}".format(pValue,error))
print("\n\n\nPrograma Terminado\n")