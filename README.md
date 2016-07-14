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

Throughout this work, we will
