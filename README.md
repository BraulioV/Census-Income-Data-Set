# Census Income Data Set
Aplicación de técnicas de aprendizaje automático al conjunto de datos de **Census Income** también conocido como _**Adult**_.

Este conjunto de datos puede encontrarse [aquí](http://archive.ics.uci.edu/ml/datasets/Census+Income "aquí")

Below in english.

## Introducción

A lo largo de este trabajo, se aplicarán diferentes técnicas y métodos para ajustar un modelo capaz de ajustar el conjunto de datos y que tenga un buen comportamiento a la hora de predecir datos futuros. En un principio, tendremos una sección de procesamiento del conjunto de datos, para poder hacerlos útiles a los algoritmos que se usarán. Como primera aproximación, al contener el conjunto de datos instancias con valores perdidos en las variables predictoras, se optará por aprender un modelo sin el conjunto de datos completo, es decir, eliminando las instancias con datos perdidos. Tras esto, utilizaremos los siguientes algoritmos para aprender un modelo _**M'**_, y seleccionar un modelo final **M** que será el que mejor se comporte.

## Modelos desarrollados

A lo largo del trabajo, se desarrollan distintos modelos que ajustarán los datos. Se analizará las ventajas de cada uno de los modelos, su complejidad, qué ventajas tiene respecto al conjunto de datos que tenemos, y el error que cometerá cada uno de los modelos:

  - **Modelo lineal**: se usará como modelo lineal la *regresión logística*.  
  - **Modelo Random Forest**
  - **Modelos de Base Radial**: en este caso, usaremos dos modelos de base radial diferentes:
    - **Support Vector Machine**
    - **KNN**
  - **Red Neuornal**

## Selección del modelo *M*

Tras desarrollar cada uno de los modelos, tendremos varios modelos *M'*, donde cada uno tendrá unas características diferentes a los demás. Tras compararlos todos, se seleccionará el modelo que mejor se comporte.

## Recuperando el conjunto de datos completo

Una vez seleccionado un modelo *M*, que será el mejor modelo de los anteriores, tiene la "carencia" de que no ha utilizado todos los datos disponibles para aprender el modelo. Por ello, se usarán modelos para ajustar las variables predictoras en las que hay valores perdidos, predecir los valores perdidos y reconstruir el conjunto de datos. Esta forma de proceder se debe a que las variables predictoras en las que hay datos perdidos, son variables categóricas, por lo que no es posible realizar otro método como sería sustituir los valores perdidos por la media de los valores, entre otros.

Una vez reconstruido el conjunto de datos, se creará un nuevo modelo _**M**_ que aprenderá y ajustará el modelo con todos los datos disponibles.

------------------------------------------------------------------------------------------------------------------------------------

Applying machine learning techniques to Census Income data set, a.k.a Adult data set.

This data set can be found [here](http://archive.ics.uci.edu/ml/datasets/Census+Income "here")

## Introduction

Throughout this work, we will apply different techniques and methods to fit the data set and create model that works well when predicting new data. First, we process the data to make it useful to the algorithms that we will use. As a first approach, due to missing values in the dataset, it will be fitted a model using a subset of all dataset of which missing values are removed. After that, it will be selected the model ___M___ which has the best fit.

## Developed models

Throughout this work, there are developed several models which will fit the dataset. There will be analyzed the advantages of each one, its complexity, which advantages each model has regarding to the dataset and the error each model has:

* _Linear model__: _logistic regression_ will be used.
* __Random Forest__
* __Radial basis models__: two different types of models will be used:
  * __Support Vector Machine__
  * __KNN__
* __Neural Network__

## Selection of the best model ___M___

After developing each model, there will be some models _M'_, where each one will have better properties to others. After comparing them, the best one will be selected.

## Getting the full dataset

Once selected the best model, ___M___, it's well known that it has been developed without using the full dataset. That's why some models will be used to fit the variables where there are missing values so we can predict them and restore the full dataset. It must be done this way because all the variables where there are missing values are categorical, so it's impossible to apply any other technique as, for example, use the average of the data.

Once restored the dataset, it will be created a new model ___M___ which will fit and learn the model with all data available.

