from ast import Try
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from os.path import join, isdir, isfile, exists, splitext

import math
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import multiprocessing as mp, time, ctypes, os
 
from re import search
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,\
                            r2_score, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, BatchNormalization,\
                                    LSTM, Dropout, GRU, SimpleRNN,\
                                    InputLayer, Conv1D, MaxPooling1D,\
                                    AveragePooling1D, Flatten
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, Adagrad, Adamax, Adadelta, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

class GlaubenDataModeling:

    def __init__(self, data, target_acronym):
        """ Constructor de la clase GlaubenDataModeling.

            Parámetros:
              - data: Lista de dataframes con los datos de cada planta para el modelado de datos.
              - target_acronym: Variable tipo String que tiene el tipo de variable a predecir (SDI,FN,None)
        """
        self.data = data
        self.target_acronym = target_acronym

        self.df_data = None
        self.X = None
        self.y = None
        self.X_normalized = None
        self.model = None
        self.history = None
        # Conjuntos de datos para entrenamiento.
        self.X_train = []
        self.X_val   = []
        self.X_test  = []

        self.y_train = []
        self.y_val   = []
        self.y_test  = []

        self.X_train_orig = []
        self.y_train_orig = []
        #variables para experimento
        self.nExp = None
        self.modelName = None
        self.learningRate = None
        self.nEpochs = None
        self.patience = None
        self.cv = False

    def selectExp(self):
        """
            Función para seleccionar los parámetros, hiperparámetros y número de experimento necesarios para realizar un entrenamiento del modelo

            Variables a seleccionar
                - nExp: Variable tipo int que contiene el número del experimento a realizar, para colocar en el nombre de los pesos.
                - modelName: Variable tipo String que contiene el nombre del modelo a utilizar
                - learningRate: Variable tipo float que contiene el learning rate a utilizar
                - nEpochs: Variable tipo int que contiene la cantidad de épocas que el entrenamiento realizará
                - patience: Variable tipo int que contiene la cantidad de paciencia que tendrá el modeo
                - cv: Variable tipo boolean que contiene True en caso de realizar cross validation
        """
        while True:
            try:
                self.nExp = int(input('Ingrese el número del experimento: '))
                break
            except:
                print("Error: Solo ingrese números.")
        modelos = ['nn', 'lstm', 'kt-sdi', 'gru', 'srnn', 'cnn-lstm']
        print("Seleccione el modelo para el experimento.")
        for i in range(len(modelos)):
            print("    "+str(i + 1)+") " + modelos[i])
        while True:
            try:
                pos = int(input("Ingrese la opción: ")) - 1
                self.modelName = modelos[pos]
                if pos < len(modelos) and pos >= 0:
                    break
            except:
                print("Error: Solo ingrese números de las opciones del menú.")
        while True:
            try:
                self.nEpochs = int(input("Ingrese el número de épocas: "))
                break
            except:
                print("Error: Ingrese solo números.")
        while True:
            try:
                self.learningRate = float(input("Ingrese el learning rate: "))
                break
            except:
                print("Error: Solo ingrese números")
        while True:
            try:
                self.patience = int(input("Ingrese la paciencia: "))
                break
            except:
                print("Error: Solo ingrese números")
        while True:
            cv = input("Desea utilizar cross validation (s/n): ")
            if cv == "s":
                self.cv = True
                break
            elif cv == "n":
                self.cv = False
                break
            else:
                print("Error: solo ingrese 's' o  'n'.")
        return

    def createModel(self, mode_name, l_rate):
        """
            Función para crear el modelo para el entrenamiento

            Parámetros
                - mode_name: Variable tipo string que contiene el nombre del modelo a utilizar en el entrenamiento.
                - l_rate: Variable tipo int que contiene el learning rate a utilziar en el entrenamiento.
        """
        opt = Adam(learning_rate=l_rate) # Adagrad, Adadelta, Adamax, Adam, RMSprop, SGD, etc.
        mae  = tf.keras.losses.MeanAbsoluteError()
        rmse = tf.keras.metrics.RootMeanSquaredError()
        mape = tf.keras.losses.MeanAbsolutePercentageError()
        _metrics = [mae, rmse, mape]
        if mode_name == 'nn':
            regLayer = l1_l2(l1=0.001, l2=0.01)
            #_l0 = Dense(units=256, input_shape=(None, 4), kernel_regularizer=regLayer, bias_regularizer=regLayer)
            _l0 = Dense(units=256, input_shape=(4,), kernel_regularizer=regLayer, bias_regularizer=regLayer, activation='relu')
            _l1 = Dense(units=128, kernel_regularizer=regLayer, bias_regularizer=regLayer, activation='relu')
            _l2 = Dense(units=64,  kernel_regularizer=regLayer, bias_regularizer=regLayer, activation='relu')
            _l3 = Dense(units=32,  kernel_regularizer=regLayer, bias_regularizer=regLayer, activation='relu')
            _l4 = Dense(units=16,  kernel_regularizer=regLayer, bias_regularizer=regLayer, activation='relu')
            _l5 = Dense(units=8,   kernel_regularizer=regLayer, bias_regularizer=regLayer, activation='relu')
            _l6 = Dense(units=1, activation='linear')
            topology = [_l0, _l1, _l2, _l3, _l4, _l5, _l6]
            self.model = tf.keras.Sequential(topology)
            self.model.compile(loss='mae', optimizer=opt, metrics=_metrics)
        elif mode_name == 'lstm':
            self.model = tf.keras.Sequential()
            self.model.add(LSTM(units=256, batch_input_shape=(None, 4, 1), return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(LSTM(units=128, return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(LSTM(units=64, return_sequences=False)) 
            self.model.add(Dropout(0.25))
            self.model.add(Dense(units=1, activation='linear'))
            self.model.compile(loss='mae', optimizer=opt, metrics=_metrics)
        elif mode_name == 'kt-sdi':
            l_rate = 0.0074
            opt = Adam(learning_rate=l_rate) # Adagrad, Adadelta, Adamax, RMSprop, SGD, etc.
            self.model = tf.keras.Sequential()
            self.model.add(InputLayer(batch_input_shape=(None, 4, 1)))
            self.model.add(LSTM(units=256, return_sequences=True))
            self.model.add(Dropout(0.1))
            self.model.add(LSTM(units=256, return_sequences=True))
            self.model.add(Dropout(0.1))
            self.model.add(LSTM(units=256, return_sequences=False)) 
            self.model.add(Dropout(0.1))
            # model.add(Flatten(name="Flatten"))
            self.model.add(Dense(units=1, activation='linear'))
            self.model.compile(loss='mae', optimizer=opt, metrics=_metrics)
        elif mode_name == 'gru':
            self.model = tf.keras.Sequential()
            self.model.add(GRU(units=256, batch_input_shape=(None, 4, 1), return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(GRU(units=128, return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(GRU(units=64, return_sequences=False)) 
            self.model.add(Dropout(0.25))
            # model.add(Flatten(name="Flatten"))
            # model.add(Dense(units=4, activation='relu'))
            self.model.add(Dense(units=1, activation='linear'))
            self.model.compile(loss='mae', optimizer=opt, metrics=_metrics)
        elif mode_name == 'srnn':
            self.model = tf.keras.Sequential()
            self.model.add(SimpleRNN(units=256, batch_input_shape=(None, 4, 1), return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(SimpleRNN(units=128, return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(SimpleRNN(units=64, return_sequences=False)) 
            self.model.add(Dropout(0.25))
            # model.add(Flatten(name="Flatten"))
            # model.add(Dense(units=4, activation='relu'))
            self.model.add(Dense(units=1, activation='linear'))
            self.model.compile(loss='mae', optimizer=opt, metrics=_metrics)
        elif mode_name == 'cnn-lstm':
            inp_shape = (None, 4, 1)
            self.model = tf.keras.Sequential()
            # "Downsampling"
            self.model.add(InputLayer(batch_input_shape=inp_shape))
            self.model.add(Conv1D(32, 2, activation="relu", padding="valid", strides=1, name="Conv1D_1"))
            self.model.add(Conv1D(64, 2, activation="relu", padding="valid", strides=1, name="Conv1D_2"))
            self.model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
            # Sección RNN
            self.model.add(LSTM(units=100, return_sequences=False, name="LSTM"))
            self.model.add(Dropout(0.2, name="Dropout_1")) # Regularización
            # Sección predictiva
            self.model.add(Flatten(name="Flatten"))
            #model.add(Dense(units=4, activation='relu', name="Dense_1"))
            self.model.add(Dense(units=1, activation='linear', name="Dense_Final"))
            self.model.compile(loss='mae', optimizer=opt, metrics=_metrics)
        return

    def normalizeData(self,normalization_name: str):
        """
            Función para normalizar la data para el entrenamiento

            Parámetros
                - normalization_name: Variable tipo string que contiene que tipo de normalización se deberá aplicar.
        """
        # TODO: Implementar clausula if para crear los objetos Scaler,
        # dependiendo del valor de normalization_name y retornamos los datos normalizados.
        if normalization_name == 'standard':
            X_scaler = StandardScaler()
            X_scaler.fit(self.X)
            X_norm = X_scaler.transform(self.X)
        elif normalization_name == 'minMax':
            X_scaler = MinMaxScaler()
            X_scaler.fit(self.X)
            X_norm = X_scaler.transform(self.X)
        return X_norm

    def trainModel(self, n_epochs: int, mode_name, b_size: int, learning_rate: float, exp_number: int, patience: int, optimizer_name: str, cv=True):
        """
          Función que se encarga de entrenar nuestro modelo neuronal. Puede implementarse
          cross-validation con el parámetro cv.

          Parámetros
            - n_epochs: Variable tipo int que contiene el número de épocas que realizará el entrenamiento. 
            - mode_name: Variable tipo String que contiene el nombre del modelo para el entrenamiento.
            - b_size: Variable tipo int que contiene el batch size.
            - learning_rate: Variable tipo int que contiene el learning rate.
            - exp_number: Variable tipo int que contiene el número del experimento.
            - patience: Variable tipo int que contiene la paciencia para el entrenamiento.
            - optimizer_name: Variable tipo string que contiene el optimizador (generalmente Adam).
          Retorna:
            - model: tf.keras....

        """
        self.createModel(mode_name=mode_name, l_rate = learning_rate)
        if mode_name == 'nn':
            self.X_train = self.X_train.reshape(-1, 4)
            self.X_test  = self.X_test.reshape(-1, 4)
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test  = self.y_test.reshape(-1 ,1)
            if cv == False:
                self.X_val = self.X_val.reshape(-1, 4)
                self.y_val = self.y_val.reshape(-1 ,1)
        else:
            self.X_train = self.X_train.reshape(-1, 4, 1)
            self.X_test  = self.X_test.reshape(-1, 4, 1)
            if cv == False:
                self.X_val = self.X_val.reshape(-1, 4, 1)

        checkpoint_name = f"EXP #{exp_number}; model {mode_name.upper()}; variable {self.target_acronym}.hdf5"
        filepath = join('C:/Users/maiko/OneDrive/Documentos/Glauben ecology/Vertientes del desierto II/Modelos/Pesos', checkpoint_name)
        print(filepath)
        our_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            patience=patience, 
                            mode='min'),
                        tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            mode='min',
                            verbose=0,
                            save_weights_only=True,
                            save_best_only=True)]

        if cv:
            kf = KFold(4, shuffle=True, random_state=123)
            iteration = 0
            models = []
            val_maes = []
            losses = []
            val_losses = []
            x_points = []
            x_ticks = [] 
            total_epocas = 0
            for (train_idx, val_idx) in kf.split(self.X_train_orig):
                print(f".::::: ENTRENANDO MODELO #{iteration + 1} :::::.")
                
                X_train, X_val = self.X_train_orig[train_idx], self.X_train_orig[val_idx]
                y_train, y_val = self.y_train_orig[train_idx], self.y_train_orig[val_idx]
                
                # Re-dimensionamos los conjuntos de train y val
                X_train, X_val = X_train.reshape(-1, 4, 1), X_val.reshape(-1, 4, 1)
                
                # Entrenamos el modelo
                history = self.model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), epochs=n_epochs, 
                        batch_size=256, verbose=2, callbacks=our_callbacks)
                
                # Agregamos el modelo a nuestra lista de modelos
                models.append(self.model)
                
                # Realizamos predicciones
                y_pred = self.model.predict(self.X_test)
                
                # Evaluamos el modelo utilizando el conjunto de pruebas
                MAE  = mean_absolute_error(self.y_test, y_pred)
                MAPE = mean_absolute_percentage_error(self.y_test, y_pred)
                MSE  = mean_squared_error(self.y_test, y_pred)
                RMSE = math.sqrt(MSE)
                R2   = r2_score(self.y_test, y_pred)
                #if MAE < min_mae:
                #  min_mae = MAE
                #  pos_mae = iteration
                print("MAE para modelo {}: {:.4f}".format(iteration, MAE))
                print("MAPE para modelo {}: {:.4%}".format(iteration, MAPE))
                print("MSE para modelo {}: {:.4f}".format(iteration, MSE))
                print("RMSE para modelo {}: {:.4f}".format(iteration, RMSE))
                print("R2 para modelo {}: {:.4f}".format(iteration, R2))
                iteration += 1
                # Probar en conjunto de TEST y calcular MAE, MAPE, RMSE, ETC
                # Guardar loss en val_losses
                #print(min_mae)


                total_epocas += len(history.history['loss'])
                losses.append(history.history['loss'])
                val_losses.append(history.history['val_loss'])
                x_points.append(np.arange(0, len(history.history['loss'])))
                x_ticks.append(np.arange(0, len(history.history['loss']), 10))
        else:
                history = self.model.fit(self.X_train, self.y_train,
                        validation_data=(self.X_val, self.y_val),
                        verbose=2,
                        epochs=n_epochs, batch_size=b_size,
                        callbacks=our_callbacks)
        
        # Implementar clausula if para los casos en los que se desee implementar CV o no.
        # self.model.fit()
        return history

    def loadWeights(self, model_weights_dir):
        """
            Función para cargar los pesos guardados de un entrenamiento

            Parámetros
                - model_weigths_dir: Variable tipo String que contiene la dirección de los pesos a cargar.

            Return
        """
        self.model.load_weights(model_weights_dir)
        return

    def selectData(self):
        """
            Función para seleccionar las columnas necesarias para el entrenamiento de los datos, en caso de ser SDI o FN, tendrá columas por defecto.
        """
        if self.target_acronym == 'SDI':
            self.X = self.df_data[['Temperatura entrada', 'Flujo de Alimentacion',
                            'Presion de entrada', 'Conductividad de entrada']]
            self.y = self.df_data['SDI Entrada RO']
        elif self.target_acronym == 'FN':
            self.X = self.df_data[['pH entrada', 'Flujo de Alimentacion',
                            'Presion de entrada', 'Conductividad de entrada']]
            self.y =  self.df_data['Flujo normalizado']
        return

    def loadData(self):
        """
            Función para subir los datos a la variable correspondiente en pandas.
        """
        self.df_data = pd.concat(self.data, ignore_index=True)
        return
    def saveModel(model_dir: str):
        """
            Función para guardar el modelo

            Parámetros
                - model_dir: Variable tipo String que contiene la dirección del modelo.
        """
        # self.model.save(model_dir)
        return

    def splitData(self, X, y, k=4):
        """
            Función para separar los datos de forma repartida para el entramiento.

            Parámetros
                - X: Arreglo tipo Pandas que contiene los datos a separar.
                - y: Arreglo tipo Pandas que contiene los datos a separar.

            Return
                Retorna 4 arreglos tipo pandas para el entrenamiento y test
        """
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        i = 1
        for (_x, _y) in zip(X, y):
            if i < k:
                X_train.append(_x)
                y_train.append(_y)
                i += 1
            else:
                X_test.append(_x)
                y_test.append(_y)
                i = 1
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test  = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test  = np.asarray(y_test, dtype=np.float32)

        return (X_train, X_test, y_train, y_test)

    def autoTrain(self):
        """
            Función para realizar un entrenamiento simi automático, donde se deberán especificar los hiperparámetros y parámetros necesarios.
        """
        self.loadData()
        self.selectData()
        K = 5
        self.selectExp()
        if str(type(self.X)) == "<class 'numpy.ndarray'>":
            self.X_train, self.X_test, self.y_train, self.y_test = self.splitData(self.X, self.y, k=K)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.splitData(self.X.values, self.y.values, k=K)
        if self.cv == False:
            K = 4
            self.X_train, self.X_val, self.y_train, self.y_val = self.splitData(self.X_train, self.y_train, k=K)
        else:
            # Variables para llevar a cabo el proceso de cross-validation
            self.X_train_orig = self.X_train.copy()
            self.y_train_orig = self.y_train.copy()
        self.history = self.trainModel(n_epochs = self.nEpochs, b_size=256,learning_rate=self.learningRate, exp_number=self.nExp, patience=self.patience, optimizer_name='adam', cv=self.cv, mode_name=self.modelName)
        return

    def doPredict(self, valor1, valor2, valor3, valor4, standard = False):
        model_input = np.hstack((valor1, valor2, valor3, valor4))
        model_input = model_input.reshape(1, 4)
        if standard:
            model_input = self.X_scaler.transform(model_input)
            model_input = model_input.reshape(1, 4, 1)
        prediction = self.model.predict(model_input)
        return prediction