import numpy as np 
import pandas as pd


file ='housing.csv'

dataframe = pd.read_csv(file, sep =',', encoding='utf-8')
print(dataframe.head(10))

x_previsores =dataframe.iloc[:,0:3].values
y_regression =dataframe.iloc[:,3].values

from sklearn.preprocessing import StandardScaler

x_previsores_scaler = StandardScaler().fit_transform(x_previsores)
y_regression_scaler = StandardScaler().fit_transform(y_regression.reshape(-1,1))

from sklearn.model_selection import train_test_split

x_train_scaler, x_test_scaler, y_train_scaler, y_test_scaler = train_test_split(x_previsores_scaler,y_regression_scaler, test_size=0.3, random_state=0)

from sklearn.neural_network import MLPRegressor
rede_neural_regressor = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', max_iter=2000, solver="adam" )

rede_neural_regressor.fit(x_train_scaler,y_train_scaler.ravel())

"""
O coeficiente de determinação, também chamado de R²,
é uma medida de ajuste de um modelo estatístico linear generalizado, 
como a regressão linear simples ou múltipla, aos valores observados 
de uma variável aleatória
"""

print('Coeficinte de determinação: {:.4f}'.format(rede_neural_regressor.score( x_test_scaler,y_test_scaler.ravel() )))

y_previsoes_scaler = rede_neural_regressor.predict(x_test_scaler)

from  metricas_erros import metricas_erros
metricas_erros(y_test_scaler, y_previsoes_scaler)

#from validacao_cruzada_rede_neural import validacao_cruzada_rede_neural
#validacao_cruzada_rede_neural(x_previsores_scaler,y_regression_scaler )


