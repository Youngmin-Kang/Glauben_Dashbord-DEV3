import email
import re
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import render, redirect
from .models import User, Prediccion
from .forms import UserForm
from re import search
from plotly.offline import plot
import unidecode
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect
from django.template import loader
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import numpy as np
import pandas as pd
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
from dotenv import load_dotenv
import plotly.graph_objects as go
from os.path import join, exists, isfile, isdir
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Create your views here.
@login_required(login_url='login')

def Glauben(request):
    
    if  request.method == "POST":
        print(request.POST) 
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        User.objects.create_user(username, email, password)  
    return render(request, "users/Glauben.html")

def login_view(request):

    return render(request, "users/login.html")

def login_Sidebar(request):

    return render(request, "users/sidebarGlauben.html")

def editInfo(request):

    return render(request, "users/editarInfo.html" , {'users': User.objects.all()})

def view_user(request, id):
    user = User.objects.get(pk=id)
    return HttpResponseRedirect(reverse('editarInfo'))

def add(request):
    if request.method=='POST':
        form = UserForm(request.POST)
        if form.is_valid():
            new_first_name = form.cleaned_data['first_name']
            new_email = form.cleaned_data['email']
            new_rol = form.cleaned_data['rol']
            
            new_user = User(
            first_name = new_first_name,
            email = new_email,
            rol = new_rol
            )   
            new_user.save()
            return render(request, 'users/add.html', {
                'form': UserForm(),
                'success': True
            })  
    else:
        form = UserForm()
    return render(request, 'users/add.html', {
        'form': UserForm()            
    })

def edit(request, id):
    if request.method == 'POST':
        user = User.objects.get(pk=id)
        form = UserForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return render(request, 'users/edit.html', {
                'form': form,
                'success': True
            })
    else:
        user = User.objects.get(pk=id)
        form = UserForm(instance=user)
    return render(request, 'users/edit.html', {
                'form': form
            })
def eliminar(request, id):
    if request.method == 'POST':
        user = User.objects.get(pk=id)
        user.delete()
    return HttpResponseRedirect(reverse('editarInfo'))

def graficoBar(request):

    return render(request, "users/estadistica.html")

def GlaubenLogin_view(request):
    
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        user = authenticate(username=email, password=password)
        if user is not None:
            login(request, user)
            print("auteticacion exito")
            return render(request, "users/sidebarGlauben.html")
        else:
            print("auteticacion fallado")            
    return render(request, "users/GlaubenLogin.html")

def prediccion(request):
    
    temp = request.POST['temp']
    conduc = request.POST['conduc']
    difer = request.POST['difer']
    flujoA = request.POST['flujoA']

    df_data = pd.read_csv("./users/data/sdi_training_dataset_4.csv")
    df_sdi = df_data[
        [
            "Temperatura entrada",
            "Conductividad de permeado",
            "Diferencial de Filtros Cartucho",
            "Flujo de Alimentacion",
        ]
    ].values
    X_scaler = MinMaxScaler()
    X_scaler.fit(df_sdi)

    #opt = Adam(learning_rate=0.001) # Adagrad, Adadelta, Adamax, Adam, RMSprop, SGD, etc.
    #mae  = tf.keras.losses.MeanAbsoluteError()
    #rmse = tf.keras.metrics.RootMeanSquaredError()
    #mape = tf.keras.losses.MeanAbsolutePercentageError()
    #_metrics = [mae, rmse, mape]
    #model = tf.keras.Sequential()
    #model.add(LSTM(units=256, batch_input_shape=(None, 4, 1), return_sequences=True))
    #model.add(Dropout(0.25))
    #model.add(LSTM(units=128, return_sequences=True))
    #model.add(Dropout(0.25))
    #model.add(LSTM(units=64, return_sequences=False)) 
    #model.add(Dropout(0.25))
    #model.add(Dense(units=1, activation='linear'))
    #model.compile(loss='mae', optimizer=opt, metrics=_metrics)
    #model.load_weights("./users/models/EXP #29.hdf5")
    model = tf.keras.models.load_model("./users/models/SDI-Model.h5", compile=False)
    model_input = np.hstack((temp, conduc, difer, flujoA))
    model_input = model_input.reshape(-1, 4)
    model_input_norm = X_scaler.transform(model_input)
    model_input_norm = model_input_norm.reshape(1, 4, 1)
    try:
        prediction = model.predict(model_input_norm)
    except Exception as e:
        print(e)
        pass
    prediction = round(prediction[0][0], 2)
    if prediction > 0 and prediction <= 3:
        estadoOperacion = "Ideal"
    elif prediction > 3 and prediction <= 4:
        estadoOperacion = "Semi-compleja"
    elif prediction > 4 and prediction <= 5:
        estadoOperacion = "Compleja"
    elif prediction > 5:
        estadoOperacion = "Inviable"
    print(prediction)
    context = {
        "temp" : temp,
        "conduc" : conduc,
        "difer" : difer,
        "flujoA" : flujoA,
        "pred" : prediction,
        "estadoOp" : estadoOperacion
    }
    #prediction = Prediccion.objects.create(pred = prediction, temp = temp,conduc = conduc, difer = difer, flujoA = flujoA, user = User.objects.get(username=request.user.username))
    return render(request, "users/sidebarGlauben.html", context)

def register(request):
    list(messages.get_messages(request))
    if request.method == "POST":
        uname = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        passwordC = request.POST["passwordC"]
        if password == passwordC:
            user = User.objects.create_user(username = email, email = email, password = password, first_name=uname)
            user.save()
            print(f'Cuenta con correo: {email} creada')
            return redirect('sidebarGlauben')
        else:
            messages.error(request, "Las contraseña debe ser la misma en ambas casillas.")
            print("mensaje añadido")
            return render(request, "users/GlaubenLogin.html")
    else:
        return render(request, "users/GlaubenLogin.html")
    

def glauben_login(request):
    list(messages.get_messages(request))
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(email," ",password)
        user = authenticate(request, username=email, password=password)
        print(user)
        if user is not None:
            login(request, user)
            return redirect('sidebarGlauben')
        else:
            messages.error(request, "Ingreso incorrecto, revise el correo o la contraseña.")
            print("mensaje añadido")

    return redirect("login1")

def glauben_logout(request):
    logout(request)
    return redirect("login1")

def fn_sidebar(request):
    return render(request, "users/FNsidebarGlauben.html")

def prediccion_fn(request):
    
    flujoR = request.POST['flujoR']
    temp = request.POST['temp']
    presion = request.POST['presion']
    conduc = request.POST['conduc']
    
    caudalNom = float(request.POST['caudalNom'])
    flujoPerm = float(request.POST['flujoPerm'])

    df_data = pd.read_csv("./users/data/fn_training_dataset_4.csv")
    df_data = df_data.iloc[:,2:]
    df_fn = df_data[
        [
            'Flujo de rechazo',
            'Conductividad de entrada',
            'Presion de entrada',
            'Temperatura entrada'
        ]
    ].values
    X_scaler = StandardScaler()
    X_scaler.fit(df_fn)

    opt = Adam(learning_rate=0.001) # Adagrad, Adadelta, Adamax, Adam, RMSprop, SGD, etc.
    mae  = tf.keras.losses.MeanAbsoluteError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    _metrics = [mae, rmse, mape]
    model = tf.keras.Sequential()
    model.add(LSTM(units=256, batch_input_shape=(None, 4, 1), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=64, return_sequences=False)) 
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mae', optimizer=opt, metrics=_metrics)
    model.load_weights("./users/models/EXP #27; model lstm; variable Flujo normalizado.hdf5")

    model_input = np.hstack((flujoR, conduc, presion, temp))
    model_input = model_input.reshape(1, 4)
    model_input_norm = X_scaler.transform(model_input)
    try:
        prediction = model.predict(model_input_norm)
    except Exception as e:
        print(e)
        pass
    prediction = round(prediction[0][0], 2)

    performance = round((prediction / caudalNom) * 100, 2)
    if performance > 100:
        performance = 100

    performance2 = round((flujoPerm / caudalNom) * 100, 2)
    if performance2 > 100:
        performance2 = 100

    print(prediction)
    context = {
        "temp" : temp,
        "conduc" : conduc,
        "presion" : presion,
        "flujoR" : flujoR,
        "pred" : prediction,
        "caudalNom" : caudalNom,
        "flujoPerm" : flujoPerm,
        "perfomance" : performance,
        "perfomance2" : performance2
    }
    #prediction = Prediccion.objects.create(pred = prediction, temp = temp,conduc = conduc, difer = difer, flujoA = flujoA, user = User.objects.get(username=request.user.username))
    return render(request, "users/FNsidebarGlauben.html", context)

def multipleSDI_data(request):
    if request.method == 'POST':
        df = ""
        file = request.FILES["myFile"]
        check = request.POST.get("limpio","false")
        encodings = ['utf-8', 'latin']
        for encode in encodings:
            try:
                df = pd.read_csv(file, encoding=encode)
                break
            except Exception as e:
                print("Exception:", e)
        try:
            df = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            print("Exception:", e)
        try:
            df = pd.read_excel(file, engine='xlrd')
        except Exception as e:
            print("Exception:", e)
        if check == "false":
            undesired_strings = ['Planta 3 - Rack 9-','Planta 1 - Rack 1-', 'Planta 1 - Filtros 1er Etapa-', 'dd','Rack2-', 'Planta 0 - Filtros Cartucho-',
                                'Planta 0 - Filtros 1era Etapa-', 'Rack1-', 'Rack2-', 'Rack 1-', 'Rack1-', 'Planta 0 - Filtros Cartucho-', 'Planta 3 - Rack 9-']
            _col_names = []
            col_names = df.columns.tolist()
            for col_name in col_names:
                for _string in undesired_strings:
                    if (_string is not None) and search(_string, col_name):
                        _col_name = col_name.replace(_string, "")
                        _col_name =  unidecode.unidecode(_col_name)
                        break
                    else:
                        _col_name = col_name
                _col_names.append(_col_name)
            df.columns = _col_names

        df.dropna()

        df_sdi = df[
        [
            "Temperatura entrada",
            "Conductividad de permeado",
            "Diferencial de Filtros Cartucho",
            "Flujo de Alimentacion",
        ]
        ]


        #Filter by gauss
        orig_cols = df_sdi.columns.tolist()
        str_colums = ["Timestamp"]
        df_columns = [i for i in orig_cols if i not in str_colums]
        result_df = df_sdi.copy(deep=True)
        if check == "false":
            for col in df_columns:
                col_df = result_df[col].copy(deep=True)
                col_mean = round(np.mean(col_df), 4)  # Obtenemos promedio
                col_std = round(np.std(col_df), 4)  # Obtenemos desviación estándar
                lower_limit = col_df > col_mean - 3 * col_std
                upper_limit = col_df < col_mean + 3 * col_std
                between_gauss = lower_limit & upper_limit
                # Esto genera que el DF no sea 'cuadrado' !
                # result_df[col] = col_df[between_gauss]
                # La siguiente línea asegura que el dataset sea simétrico !
                delete_indexes = between_gauss[~between_gauss].index.values
                #print(len(delete_indexes))
                #print(col)
                result_df = result_df.drop(delete_indexes, axis=0)
        #end of filter


        df_data = pd.read_csv("./users/data/sdi_training_dataset_4.csv")
        df_sdi = df_data[
            [
                "Temperatura entrada",
                "Conductividad de permeado",
                "Diferencial de Filtros Cartucho",
                "Flujo de Alimentacion",
            ]
        ].values
        X_scaler = MinMaxScaler()
        X_scaler.fit(df_sdi)

        model = tf.keras.models.load_model("./users/models/SDI-Model.h5", compile=False)
        y_pred = np.array([], dtype=np.float32)
        try:
            df_data = result_df
            for (idx, row) in result_df.iterrows():
                print(row)
                temperatura_entrada = row["Temperatura entrada"]
                conductividad_permeado = row["Conductividad de permeado"]
                dif_filtros_cartuchos = row["Diferencial de Filtros Cartucho"]
                flujo_alimentacion = row["Flujo de Alimentacion"]
                # Creamos el arreglo de entrada para el modelo !
                model_input = np.hstack(
                    (
                        temperatura_entrada,
                        conductividad_permeado,
                        dif_filtros_cartuchos,
                        flujo_alimentacion,
                    )
                )
                model_input = X_scaler.transform(model_input.reshape(-1, 4))
                model_input = model_input.reshape((1, 4, 1))
                model_input = model_input.astype("float32")
                curr_prediction = model.predict(model_input)
                y_pred = np.append(y_pred, curr_prediction[0], axis=0)
        except Exception as e:
            print("Ha surgido un error !")
            print(e)
            print()
        print(y_pred)
        sdiMin = min(y_pred)
        sdiMax = max(y_pred)
        sdiProm = y_pred.mean()

        cantDatos = len(y_pred)
        ejeX = []
        for i in range(cantDatos):
            ejeX.append(i+1)
        graphs = []
        graphs.append(
            go.Scatter(x=ejeX, y=y_pred, mode='lines', name='Line y1')
        )
        layout = {
            'xaxis_title': 'X',
            'yaxis_title': 'Y',
            'height': 420,
            'width': 560,
        }

        plot_div = plot({'data': graphs, 'layout': layout}, 
                output_type='div')

        context = {
            "sdiMin" : sdiMin,
            "sdiMax" : sdiMax,
            "sdiProm" : sdiProm,
            "y_pred" : y_pred,
            "ejeX" : ejeX,
            "cantDatos" : cantDatos,
            "plot_div": plot_div
        }
    #prediction = Prediccion.objects.create(pred = prediction, temp = temp,conduc = conduc, difer = difer, flujoA = flujoA, user = User.objects.get(username=request.user.username))
        #print("if")
        #return render(request, "users/multipleSDI.html", context)
        return render(request, "users/multipleSDI.html", context)
    return render(request, "users/multipleSDI.html")