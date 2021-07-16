"""## imports"""
import os
import ast
import traceback
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from numpy import asarray
from sklearn import metrics
from tensorflow import keras
from skopt import gp_minimize
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from skopt.utils import use_named_args
from skopt.plots import plot_objective
from skopt.plots import plot_convergence
from skopt.space import Real, Categorical
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective_2D
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from masksembles.keras import Masksembles2D, Masksembles1D
from sklearn.metrics import confusion_matrix, precision_recall_curve

print("Starting the experiment.....")


def get_cifar100_data(label_index):
    '''
    this function get the cifar data from the keras api and split it to
    equal part of dataset containing 5 classes
    :param label_index: the index label that indicate the division
    :return: 2 dataframe that containing the ndarray of the image and the label
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    dict_data = {'data': [], 'label': []}

    for i in range(len(x_train)):
        dict_data['data'].append(x_train[i])
        dict_data['label'].append(y_train[i][0])
    for i in range(len(x_test)):
        dict_data['data'].append(x_test[i])
        dict_data['label'].append(y_test[i][0])
    df = pd.DataFrame.from_dict(dict_data)
    list_range = list(range((5 * label_index) - 5, 5 * label_index))
    df_filter = df[df['label'].isin(list_range)]
    df_filter = df_filter.sort_values(by=['label'])
    df_filter = df_filter.iloc[::6]
    df_filter = df_filter.iloc[::1]

    df_filter['label'] = df_filter['label'].astype(str)
    df_filter.label = pd.Categorical(df_filter.label)
    df_filter['code'] = df_filter.label.cat.codes

    df_filter = df_filter.drop(columns=['label'])
    df_filter = df_filter.rename(columns={"code": "label"})
    return df_filter[['data']], df_filter[['label']]


def cast_img(list_img_path):
    '''
    this function get the image path list and open each image and convert it to ndarray of RGB
    :param list_img_path: the list of image path of a certain dataset
    :return: A list of ndarray of RGB in size (32,32,3)
    '''
    list_rgb = []
    basewidth = 32
    # load the image and convert into numpy array
    for row in list_img_path:
        img = Image.open(row)
        # resize
        img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
        numpydata = asarray(img)
        list_rgb.append(numpydata)
    return list_rgb


def literal_row(row):
    '''
    this function received the dataframe containing the data
    and cast the string representation of list to ndarray
    :param row: the riw containing the data
    :return: a ndarray of the image
    '''
    return np.array(ast.literal_eval(row))


def get_data_csv(str_data):
    '''
    this function get the str_data representation of the data that needed to be read and
    return dataframe that resuced according to the condition of the name
    :param str_data: the name of the data
    :return: a dataframe of data and of label according to the name from the csv
    '''
    df = pd.read_csv('Csv_Data/' + str_data + '.csv')
    df['data'] = df['data'].apply(literal_row)
    if 'lable' in df.columns:
        df = df.rename(columns={"lable": "label"})
    df['label'] = df['label'].astype(str)
    df.label = pd.Categorical(df.label)
    df['code'] = df.label.cat.codes

    df = df.drop(columns=['label'])
    df = df.rename(columns={"code": "label"})

    return df[['data']], df[['label']]


def data_preprocessing(flag, csv=False):
    '''
    this is the main function that get the dataframe from the csv from kaggle or the keras api
    according to the string
    :param flag: the flag string of the name of the dataframe
    :param csv: a boolean flag that indicate if the data is a csv or not
    :return: a dataframe of the data and the label
    '''
    if csv:
        return get_data_csv(flag)
    else:
        if 'cifar' in flag:
            label_index = int(flag.split("_")[1])
            return get_cifar100_data(label_index)
    return None


def covert_data_np(x_train, x_test):
    '''
    this function convert a list to nparray in dimension (32,32,3) and normalize the data
    :param x_train: the x_train
    :param x_test:  the x_test
    :return: the ndarray normalize data
    '''
    x_train = np.stack(list(x_train['data']), axis=0)
    x_test = np.stack(list(x_test['data']), axis=0)
    # normalize
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    return x_train, x_test


def covert_label_np(y_train, y_test, num_classes):
    '''
    this function convert the label dataset to categorical one hot vector
    :param y_train: the y_train
    :param y_test: the y_test
    :param num_classes: the number of classes
    :return: the fix y_train and y_test
    '''
    np.asarray(list(y_train['label']))
    np.asarray(list(y_test['label']))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return y_train, y_test


# ranges of possible parameters in order to tune the model
dim_learning_rate = Real(low=1e-3, high=1e-1, prior='log-uniform', name='learning_rate')
dim_activation = Categorical(categories=['relu', 'elu'], name='activation')
optimizer = Categorical(categories=['adam', 'sgd', 'RMSprop'], name='optimizer')
dimensions = [dim_learning_rate, dim_activation, optimizer]


def create_model(learning_rate, activation, optimizer, input_shape):
    '''
    function that creates the model according to some arguments
    :param learning_rate: the learning rate
    :param activation: the activation
    :param optimizer:  the optimizer
    :param input_shape: the input shape (32,32,3)
    :return: the model Masksembles
    '''
    model = Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation=activation))
    model.add(Masksembles2D(4, 2.0))  # adding Masksembles2D
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # adding Masksembles2D

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation=activation))
    model.add(Masksembles2D(4, 2.0))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # adding Masksembles2D

    model.add(layers.Flatten())

    model.add(Masksembles1D(4, 2.))
    model.add(layers.Dense(num_classes, activation="softmax"))

    opt = keras.optimizers.Adam()
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    if optimizer == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def customize_data(data_x, data_y):
    '''
    reorganize the data
    :param data_x:
    :param data_y:
    :return: the data_x and data_y
    '''
    len_data = len(data_x)
    while len_data % 4 != 0:
        len_data = len_data - 1
    return data_x[:len_data], data_y[:len_data]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, activation, optimizer):
    """
    Hyper-parameters: the function executes the internal CV, and search for the best parameters
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    print("in fitness")
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('activation:', activation)
    print('optimizer:', optimizer)
    print()

    # Internal Cross validation For the Hyper Parameters
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    average_accuracy = 0
    for train_index, val_index in kf.split(X_train_val, Y_train_val):
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = Y_train_val.iloc[train_index], Y_train_val.iloc[val_index]

        X_train, y_train = customize_data(X_train, y_train)
        X_val, y_val = customize_data(X_val, y_val)

        X_train, X_val = covert_data_np(X_train, X_val)

        y_train, y_val = covert_label_np(y_train, y_val, num_classes)

        # Create the neural network with these hyper-parameters.
        if flag_improved:
            model = create_vgg_mask_model(num_classes=num_classes,
                                          learning_rate=learning_rate,
                                          activation=activation,
                                          input_shape=(32, 32, 3),
                                          optimizer=optimizer)
        else:
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 input_shape=(32, 32, 3),
                                 optimizer=optimizer)
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=10,
                            batch_size=128,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping]
                            )
        accuracy = history.history['val_accuracy'][-1]
        average_accuracy += accuracy

        del model
        K.clear_session()

    average_accuracy = average_accuracy / 3
    accuracy = average_accuracy
    print()
    print("Average Accuracy: {0:.2%}".format(accuracy))
    print()

    global best_accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy

    return -accuracy


def document_hyperparameters(search_result, index_cv=0):
    '''
    this part is also taken from the github, and it plots more information
    about the process of the optimization
    :param search_result: the search result parameters
    :param index_cv: the index cv
    :return:
    '''
    try:
        plt.clf()
        if not os.path.exists("Hyperparameters_Optimization/" + str(index_cv)):
            os.mkdir("Hyperparameters_Optimization/" + str(index_cv))
        plot_convergence(search_result)
        plt.savefig("Hyperparameters_Optimization/" + str(index_cv) + "/Converge.png", dpi=400)

        print(f" search result {search_result.x}")
        print(f"The best fitness value associated with these hyper-parameters {search_result.fun}")

        fig = plot_objective_2D(result=search_result,
                                dimension_identifier1='learning_rate',
                                dimension_identifier2='optimizer',
                                levels=50)
        plt.savefig("Hyperparameters_Optimization/" + str(index_cv) + "/Lr_optimizer.png", dpi=400)

        fig = plot_objective_2D(result=search_result,
                                dimension_identifier1='learning_rate',
                                dimension_identifier2='activation',
                                levels=50)
        plt.savefig("Hyperparameters_Optimization/" + str(index_cv) + "/Lr_activation.png", dpi=400)

        # create a list for plotting
        dim_names = ['learning_rate', 'activation', 'optimizer']
        plot_objective(result=search_result, dimensions=dim_names)
        plt.savefig("Hyperparameters_Optimization/" + str(index_cv) + "/all_dimen.png", dpi=400)
        plot_evaluations(result=search_result, dimensions=dim_names)
    except:
        print("error in document hyperparameters")
        traceback.print_exc()


def calculate_results(best_model, X_test, Y_test):
    '''
    the function fits and predicts by using the best model and documenting the results
    in a CSV for the report
    :param best_model: the the model with parameters
    :param X_test: the X_test
    :param Y_test: the Y_test
    :return: the result of the best model on the cv
    '''
    results = {}

    X_test, Y_test = customize_data(X_test, Y_test)

    y_pred = best_model.predict(X_test)

    y_pred_label_int = np.argmax(y_pred, axis=1)
    y_test_label_int = np.argmax(Y_test, axis=1)

    y_pred_max_value = (y_pred == y_pred.max(axis=1)[:, None]).astype(float)

    number_of_classes = [i for i in range(Y_test.shape[1])]
    # In case something goes wrong we don't want the program to exit and stop running
    try:
        results['accuracy_score'] = metrics.accuracy_score(y_test_label_int, y_pred_label_int)
    except:
        results['accuracy_score'] = "***"
        traceback.print_exc()
    try:
        results['precision_score'] = metrics.precision_score(Y_test, y_pred_max_value, average='macro',
                                                             zero_division='warn')
    except:
        results['precision_score'] = "***"
        traceback.print_exc()
    try:
        results['roc_auc_score'] = metrics.roc_auc_score(Y_test, y_pred, multi_class='ovr')
        traceback.print_exc()
    except:
        results['roc_auc_score'] = "***"
        traceback.print_exc()

    try:
        matrix = confusion_matrix(y_test_label_int, y_pred_label_int)
        FP = (matrix.sum(axis=0) - np.diag(matrix)).astype(float)
        FN = (matrix.sum(axis=1) - np.diag(matrix)).astype(float)
        TP = (np.diag(matrix)).astype(float)
        TN = (matrix.sum() - (FP + FN + TP)).astype(float)

        results['fpr'] = sum(FP / (FP + TN)) / len(number_of_classes)
        results['tpr'] = sum(TP / (TP + FN)) / len(number_of_classes)
        results['auc_score'] = metrics.roc_auc_score(Y_test, y_pred)
    except:
        results['fpr'] = "***"
        results['tpr'] = "***"
        traceback.print_exc()
    try:
        pr_curve = 0
        for i in number_of_classes:
            precision, recall, _ = precision_recall_curve(Y_test[:, i], y_pred[:, i])
            pr_curve += metrics.auc(recall, precision)
        results['pr_curve'] = pr_curve / len(number_of_classes)
    except:
        results['pr_curve'] = "***"
        traceback.print_exc()

    # check the 1000 instances
    if len(X_test) < 1000:
        X_test_1000 = X_test
    else:
        X_test_1000 = X_test[:1000]
    t_start_test = time()
    best_model.predict(X_test_1000)
    t_end_test = time() - t_start_test
    results['inference_time'] = t_end_test
    return results


def convert_params_to_string(opt_par):
    '''
    this function convert the parameters to a string
    :param opt_par: the parameters
    :return: the string parameters
    '''
    learning_rate = opt_par[0]
    activation = opt_par[1]
    optimizer = opt_par[2]
    string_result = "Learning Rate: " + str(learning_rate) + "\n" + "Activation:  " + str(
        activation) + "\n" + "Optimizer: " + str(optimizer)

    return string_result


def create_table_csv(algorithm_name, dataset_name, df):
    """
    this function that puts the data with the results of each CV
    into a csv and saves it on the disk
    :param algorithm_name: the current algorithm cnn name
    :param dataset_name: the current dataset name
    :param df: the dataframe with the results
    :return:
    """
    df.to_csv(dataset_name + "_" + algorithm_name + ".csv", index=False)


def create_vgg_model(num_classes):
    '''
    this function create the VGG (Part 3 well-known algorithm) model
    :param num_classes: the number of classes in the dataset
    :return: the VGG model
    '''
    pre_trained_model = VGG19(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False
    model = Sequential()
    model.add(pre_trained_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


def create_vgg_mask_model(num_classes, learning_rate, activation, optimizer, input_shape):
    '''
    this function create the improve model (part 2 improvement) by combaining the VGG model
    with Masksembles
    :param num_classes: the number of classes in the dataset
    :param learning_rate: the learning_rate
    :param activation: the activation
    :param optimizer: the optimizer
    :param input_shape: the input shape (32,32,3)
    :return:
    '''
    pre_trained_model = VGG19(input_shape=input_shape, include_top=False, weights="imagenet")
    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False

    model = Sequential()
    model.add(pre_trained_model)

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation=activation, padding='same'))
    model.add(Masksembles2D(4, 2.0))  # adding Masksembles2D
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))  # adding Masksembles2D

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation=activation, padding='same'))
    model.add(Masksembles2D(4, 2.0))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))  # adding Masksembles2D

    model.add(layers.Flatten())

    model.add(Masksembles1D(4, 2.))

    opt = keras.optimizers.Adam()
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    if optimizer == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


early_stopping = EarlyStopping(monitor='val_loss', patience=3)

list_data_name = ['cifar_1', 'cifar_2','cifar_3', 'cifar_4', 'cifar_5',
                  'cifar_6', 'cifar_7', 'cifar_8', 'cifar_9', 'cifar_10',
                  'butterfly1', 'butterfly2', 'butterfly3', 'fruits1','fruits2',
                  'fruits3', 'fruits4', 'monkey', 'flowers', 'intel']




input_shape = (32, 32, 3)

list_dict_model = [{'flag_vgg': False, 'algorithm_name': 'Mask_parameters_for_tomer', 'flag_improved': False},
                   {'flag_vgg': False, 'algorithm_name': 'Mask_VGG_Improved', 'flag_improved': True},
                   {'flag_vgg': True, 'algorithm_name': 'VGG', 'flag_improved': False}
                   ]

# POSITION [0] = Mask,
# POSITION [1] = improved algorithm mask with VGG,
# POSITION [2] = well known algorithm VGG
# !!!Change manually vgg or mask or improved!!!

flag_vgg = list_dict_model[0]['flag_vgg']
algorithm_name = list_dict_model[0]['algorithm_name']
flag_improved = list_dict_model[0]['flag_improved']

for name in list_data_name:
    dataset_name = name
    print("Data set name: " + dataset_name)
    try:
        if 'cifar' in name:
            list_img, list_label = data_preprocessing(name, False)
        else:
            list_img, list_label = data_preprocessing(name, True)

        num_classes = len(Counter(list(list_label['label'])))
        print(Counter(list(list_label['label'])))

        X = list_img
        Y = list_label

        # the function implements the nested cross validation and export the results
        # building the CSV we will export later for the report
        cv_number = [i for i in range(1, 11)]
        algorithm = [algorithm_name for i in range(10)]
        dataset = [dataset_name for i in range(10)]
        final_df = pd.DataFrame(dataset, columns=['Dataset Name'])
        final_df['Algorithm Name'] = algorithm
        final_df['Cross Validation'] = cv_number
        hyper_params_values = []
        acc = []
        TPR = []
        FPR = []
        precision = []
        AUC = []
        PR_Curve = []
        training_time = []
        inference_time = []

        # external loop
        # Full Nested Cross Validation
        kf_external = KFold(n_splits=10, shuffle=True, random_state=42)
        index_cv = 0
        results = {}
        all_time_loop = time()
        for train_val_index, test_index in kf_external.split(X, Y):
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            Y_train_val, Y_test = Y.iloc[train_val_index], Y.iloc[test_index]
            best_accuracy = 0.0

            # internal loop
            if not flag_vgg:
                search_result = gp_minimize(func=fitness,
                                            dimensions=dimensions,
                                            acq_func='EI',  # Expected Improvement.
                                            n_calls=50,
                                            n_jobs=-1,
                                            x0=[1e-3, 'relu', 'adam'])

                document_hyperparameters(search_result, index_cv=index_cv)
                index_cv += 1
                print('CV index: {}'.format(index_cv))
                results[tuple(search_result.x)] = best_accuracy

                # extract the best parameters from the internal CV
                opt_par = search_result.x
                learning_rate = opt_par[0]
                activation = opt_par[1]
                optimizer = opt_par[2]

                # create the best model based on the params
                if flag_improved:
                    best_model = create_vgg_mask_model(num_classes=num_classes,
                                                       learning_rate=learning_rate,
                                                       activation=activation,
                                                       optimizer=optimizer,
                                                       input_shape=(32, 32, 3))
                else:
                    best_model = create_model(learning_rate=learning_rate,
                                              activation=activation,
                                              optimizer=optimizer,
                                              input_shape=(32, 32, 3))
            else:
                best_model = create_vgg_model(num_classes)
            X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(X_train_val, Y_train_val, test_size=0.15,
                                                                              random_state=0)

            X_train_fit, y_train_fit = customize_data(X_train_fit, y_train_fit)
            X_val_fit, y_val_fit = customize_data(X_val_fit, y_val_fit)

            X_train_fit, X_val_fit = covert_data_np(X_train_fit, X_val_fit)
            y_train_fit, y_val_fit = covert_label_np(y_train_fit, y_val_fit, num_classes)

            X_train_val, X_test = covert_data_np(X_train_val, X_test)
            Y_train_val, Y_test = covert_label_np(Y_train_val, Y_test, num_classes)

            t_start_train = time()

            history = best_model.fit(X_train_fit, y_train_fit, epochs=20, batch_size=128,
                                     validation_data=(X_val_fit, y_val_fit), callbacks=[early_stopping])

            t_end_train = time() - t_start_train
            training_time.append(t_end_train)
            params_string = "params_string"
            if not flag_vgg:
                params_string = convert_params_to_string(opt_par)
            hyper_params_values.append(params_string)
            results = calculate_results(best_model, X_test, Y_test)
            acc.append(results['accuracy_score'])
            TPR.append(results['tpr'])
            FPR.append(results['fpr'])
            precision.append(results['precision_score'])
            AUC.append(results['roc_auc_score'])
            PR_Curve.append(results['pr_curve'])
            inference_time.append(results['inference_time'])
            print("total time for loops: {}".format(str(time() - all_time_loop)))
        final_df['Hyper-Parameters Values'] = hyper_params_values
        final_df['Accuracy'] = acc
        final_df['TPR'] = TPR
        final_df['FPR'] = FPR
        final_df['Precision'] = precision
        final_df['AUC'] = AUC
        final_df['PR-Curve'] = PR_Curve
        final_df['Training Time'] = training_time
        final_df['Inference Time'] = inference_time
        create_table_csv(algorithm_name, dataset_name, final_df)
    except Exception as e:
        print("ERROR in dataset : {}".format(name))
        traceback.print_exc()
        continue
