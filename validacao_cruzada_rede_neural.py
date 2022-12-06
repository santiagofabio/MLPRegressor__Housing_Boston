def validacao_cruzada_rede_neural(previsores_scaler,classe_alvo):

     from sklearn.neural_network import MLPRegressor
     from sklearn.model_selection import KFold
     from sklearn.model_selection import cross_validate, cross_val_score
     
     rede_neural   =MLPRegressor(hidden_layer_sizes=(14,14),
                            activation="logistic", solver ="adam"  , max_iter = 2000,
                            tol = 0.0001,random_state=0,verbose =True )
     
     resultados_medios =[]
     for i in range(0,30):
          kfold = KFold(n_splits=10, shuffle=True, random_state=i)
          resulado = cross_val_score(rede_neural,previsores_scaler,classe_alvo.ravel(), cv =kfold )     
          resultados_medios.append(resulado.mean()) 
    
     import pandas as pd
     import seaborn as sns
     import matplotlib.pyplot as plt
     nome_modelo ='rede_neural'
     dataframe = pd.DataFrame({nome_modelo:resultados_medios})
     dataframe.to_csv('rede_neural_regressao.csv',sep=';', encoding='utf-8', index=False)
     sns.kdeplot(data = dataframe,x= nome_modelo, label='Distribution rede neural ')
     plt.legend(loc ='best')
     plt.tight_layout()
     plt.title('RN_cross_validatio')
     plt.savefig('rede_neural_cross_validation.jpg', dpi =300, format = 'jpg')
     plt.show() 
     
     return(0)