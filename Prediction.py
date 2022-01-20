from keras.models import Sequential, model_from_json

import pandas as pd

from keras.layers import Dense    #for Dense layers
from keras.layers import BatchNormalization #for batch normalization
from keras.layers import Dropout            #for random dropout
from keras.models import Sequential #for sequential implementation
from keras.optimizers import adam_v2
from keras import regularizers      #for l2 regularization
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


def findFight(df, fighter1, fighter2):
    ind = df.axes[0]

    for i in range(len(df['R_fighter'])):
        d = ind[i]

        if df['R_fighter'][d] == fighter1 and df['B_fighter'][d] == fighter2:
            return [i,0]
        elif df['R_fighter'][d] == fighter2 and df['B_fighter'][d] == fighter1:
            return [i,1]
    return [0,0]



def load_model(X):
    learning_rate=0.0001
    activation='tanh'
    opt = adam_v2.Adam(learning_rate=learning_rate)
    n_cols = X.shape[1]
    input_shape = (n_cols,)
    model = Sequential()
    model.add(Dense(128,
                    activation=activation,
                    input_shape=input_shape,
                    activity_regularizer=regularizers.l2(1e-5)))
    model.add(Dropout(0.50))

    model.add(Dense(64,
                    activation=activation,
                    activity_regularizer=regularizers.l2(1e-5)))
    model.add(Dropout(0.50))
    model.add(Dense(2, activation='tanh'))
   # model = model_from_json(open('model_architecture.json').read())

    model.load_weights('model_weights.h5')

    model.compile(optimizer=opt,
                  loss="mean_absolute_error",
                  metrics=['mse', "mape"])
    return model



def LGBM(fighter1, fighter2):


    data_df = pd.read_csv('data.csv')
    df = data_df.dropna()
    columns = df.select_dtypes(include='object').columns
    index = findFight(df,fighter1,fighter2)
    if index[0] == 0:
        return [0]
    df.drop(columns=['R_fighter', 'B_fighter', 'Referee', 'date', 'location', 'weight_class'], inplace=True)
    df.select_dtypes(include='object')
    map_stance = {'Orthodox': 0, 'Switch': 1, 'Southpaw': 2, 'Open Stance': 3}
    df['B_Stance'] = df['B_Stance'].replace(map_stance)
    df['R_Stance'] = df['R_Stance'].replace(map_stance)

    map_winner = {'Red': 0, 'Blue': 1, 'Draw': 2}
    df['Winner'] = df['Winner'].replace(map_winner)
    df.drop(columns=df.select_dtypes(include='bool').columns, inplace=True)
    df['Winner'].unique()
    df = df[df['Winner'] != 2]
    X = df.drop(columns=['Winner'])

    # fight1 = [5, 0, 4, 0, 8, 27, 0, 188, 198, 205, 0, 20, 0, 0, 20, 1, 193, 293, 193]
    # fight2 = [5, 0, 11, 0, 4, 20, 0, 173, 165, 145, 0, 2, 0, 4, 11, 0, 183, 183, 145]
    # fight3 = [5, 0, 10, 0, 1, 15, 1, 170, 170, 135, 0, 5, 0, 3, 19, 0, 170, 180, 135]
    # fight4 = [3, 0, 6, 0, 1, 18, 0, 178, 178, 155, 0, 3, 0, 9, 23, 2, 173, 178, 155]
    # fight5 = [3, 2, 0, 0, 8, 21, 0, 188, 193, 205, 0, 1, 0, 2, 13, 0, 193, 198, 205]
    # X.info()
    # print(X.head())
    #
    # c=X.columns
    # print(c)
    # print('\n\nlength is :' + str(len(c)))
    # df1 = pd.DataFrame(np.array([fight1, fight2, fight3, fight4, fight5]), columns=X.columns)
    import pickle
    clf = pickle.load(open("model.pkl", "rb"))
    ModelK = load_model(X)
   # clf = lgb.Booster(model_file='lgbm.txt')

#    y_pred = clf.predict(X[index])
#    a = X[index]
#    s = X.rows[:]

  #  w= [[0]]

    w = [X.iloc[index[0]]]
    y_pred = clf.predict(w)
    asd = X.loc[[index[0]]]
    Proba = ModelK.predict(asd)
    a = Proba[0]
    b= a[0]
    return [True,y_pred[0], index[1], w[0], Proba[0]]

