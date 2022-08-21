# from ssl import Options # esta línea no es necesaria
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import base64
import time
from PIL import Image
im = Image.open("image.jpg")
st.set_page_config(
    page_title="Iberian Archaeological Mineral Classification",
    page_icon=im,
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/DASA39",
        'About': """ 
                   
         If you're seeing this, I would really appreciate your contribution! If you find bugs or can provide some order, please reach out or create an issue on our 
         [GitHub](https://github.com/DASA39) repository. If you find that this interface doesn't do what you need it to, you can create an feature request 
         at our repository or better yet, contribute a pull request of your own. You can find me at dasa39 [at] gmail.com
    
         More documentation and contribution details are at our [GitHub Repository](https://github.com/DASA39).
        
         This app is the result of a PhD ongoing project titled: Exploring complexity through amber and variscite.
         Computational archaeology and geoarchaeological data in the Late Prehistory of Iberia.
         A project developed by:
         
        -Daniel Sánchez-Gómez
        at University of Seville
        """
    })

timestr = time.strftime("%Y%m%d")
model = load_model('deployment_14082022')

OPTIONS = ['Predict', 'Project', 'Model Metadata', 'Model Development']

# Función para usar el modelo (predicciones)


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]

    return predictions

# Función para descargar el csv generado


def csv_downloader(data):
    csvfile = data.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "predictions_{}_.csv".format(timestr)
    st.markdown("#### Download File ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!</a>'
    st.markdown(href, unsafe_allow_html=True)

# Función principal


def run():

    #st.title("Archaeological stone classification project")
    #file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    page = st.sidebar.radio('Navigation', OPTIONS)

    if page == 'Predict':

        st.title("Archaeological p-XRF mineral data classification Model (BETA)")
        file_upload = st.file_uploader(
            "Upload csv file for predictions", type=["csv"])

        if file_upload is not None:

            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)
            csv_downloader(predictions)
        st.sidebar.write("""
         ## About
          A Machine Learning model has been developed and trained to predict the mineral Group and Sub group to which a sample obtained from p-XRF belongs.
          In order to use the model, tabular data is required. (see [template](https://github.com/DASA39))
          Once a csv file has been upload, the model will predict and display the results in a new column called Label as well
          as the probability with which the algorithm scores its prediction.
          """)

    elif page == 'Project':
        st.header("Exploring complexity through amber and variscite. Computational archaeology and geoarchaeological data in the Late Prehistory of the Iberian peninsula. (4th-to 2nd millennia BC)")
        st.sidebar.image('image.jpg')
        abstract = (""" 
         
         ## Abstract:
         
Since the expansion of the Neolithic across the Mediterranean, with the arrival and
consolidation of village life and agriculture, long-distance exchange experiences 
an unusual growth, and exotic items become an important means of displaying new 
roles and social differences within and between communities. The use and association 
of exotic items with specific individuals increases as social complexity grows and 
it becomes more necessary to enhance social differences and exhibit the status of 
the bearer. The study of the geographical origin and the Spatio-temporal distribution 
patterns of exotic raw materials and their products are trending topics in European 
archaeological research since they are considered key to the understanding of social 
interaction and the mobility patterns of individuals and /or goods at different scales. 
Amber and variscite like minerals are two of the most used materials for the 
elaboration of body adornments in prehistory that have both, an important weight in 
the archaeological record and a relevant research tradition in the Iberian Peninsula 
context. Thanks to the development of novel chemical analytical techniques, which are 
both portable and non-destructive, in the last ten years it has been possible to record
data of thousands of items of personal adornment made out of amber and variscite-like 
minerals from more than 900 archaeological sites on the Iberian Peninsula through
different projects, which represents a first-rate experimental data set for the study of 
these subjects. Despite the existence of such relevant data sets, to date, there are both 
methodological and theoretical challenges in extracting knowledge from this type of resources
due to the lack of comprehensive studies with a data-driven approach. Regarding the study of 
raw materials like amber or variscite-like minerals, it is still necessary to make exhaustive
inventories of Iberian sources, mineralogical characterisation of items, and systematisation 
of scattered and unpublished data among other urgent tasks that will improve or reconsider 
provenance models used to explain the socio-economic dynamics in late prehistory. Through 
the use of different techniques of Computational archaeology such as Data mining and 
Machine Learning, the aim of this doctoral program is to explore a data driven approach to 
solve some of the main methodological challenges in the study of the socio-economic complexity 
in the late prehistory of the Iberian Peninsula and develop an Open Access approach for 
publication of results in accordance with the necessity of digitalization of humanities.

### Aim:

To explore the socio-economic dynamics in Late Prehistory of the Iberian Peninsula associated with 
the use of body adornments of Amber and green stones through a data-driven approach that allows the 
development of open-access computational tools for the study of social complexity in prehistory.

 #### objectives:

* To Develop a Data engineering pipeline to collect, explore and create geoarchaeological data sets
for the study of body ornaments of different minerals in the Iberian Peninsula.

* To Develop machine learning-based models of mineral composition data to facilitate provenance and 
distribution maps of stones and amber in Late Prehistory of Iberia.

* To build  data sets of FTIR spectra of Iberian amber to be used in the development of machine learning-based models.

* Characterise the Iberian sources to certify that the FTIR spectra of these amber deposits are 
different from those of already known Baltic, Sicilian or Cantabrian deposits used during Late Prehistory

* To Develop open-access applications for the public use of the tools developed. 






         """)
        st.sidebar.write("""
         ## About

### 

Exploring complexity through amber and variscite. Computational archaeology and
geoarchaeological data in the Late Prehistory of the Iberian peninsula. 
is an ongoing PhD project developed in the [University of Seville](http://institucional.us.es/doctorhistoria/)


The aim of this doctoral program is to explore a data-driven
approach to solving some of the main methodological challenges in the study of the
socio-economic complexity in the late prehistory of the Iberian Peninsula and develop
Open Access alternatives for the publication of results in accordance with the necessity of
digitalization of humanities.""")
        st.markdown(abstract, unsafe_allow_html=False)

    elif page == 'Model Metadata':

        st.sidebar.write("""
         ## About""")
        metadata = (
            """ Model: 

               LGBMClassifier(bagging_fraction=0.7, bagging_freq=6, boosting_type='gbdt',
               class_weight='balanced', colsample_bytree=1.0,
               feature_fraction=0.5, importance_type='split', learning_rate=0.1,
               max_depth=-1, min_child_samples=66, min_child_weight=0.001,
               min_split_gain=0.4, n_estimators=90, n_jobs=-1, num_leaves=90,
               objective=None, random_state=123, reg_alpha=0.0005,
               reg_lambda=0.1, silent='warn', subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)

Pipeline:

Pipeline(memory=None,
          steps=[('dtypes',
                  DataTypes_Auto_infer(categorical_features=[],
                                       display_types=True, features_todrop=[],
                                       id_columns=[],
                                       ml_usecase='classification',
                                       numerical_features=[], target='target',
                                       time_features=[])),
                 ('imputer',
                  Simple_Imputer(categorical_strategy='not_available',
                                 fill_value_categorical=None,
                                 fill_value_numerical=None,
                                 numeric_strat...
                                 colsample_bytree=1.0, feature_fraction=0.5,
                                 importance_type='split', learning_rate=0.1,
                                 max_depth=-1, min_child_samples=66,
                                 min_child_weight=0.001, min_split_gain=0.4,
                                 n_estimators=90, n_jobs=-1, num_leaves=90,
                                 objective=None, random_state=123,
                                 reg_alpha=0.0005, reg_lambda=0.1, silent='warn',
                                 subsample=1.0, subsample_for_bin=200000,
                                 subsample_freq=0)]],
          verbose=False)




-------
Type:        Pipeline

String form:

Pipeline(steps=[('dtypes',
           DataTypes_Auto_infer(ml_usecase='classification',
           <...>           n_estimators=150, n_jobs=-1,
           random_state=123)]])

Length:      24

Docstring:  

Pipeline of transforms with a final estimator.

Sequentially apply a list of transforms and a final estimator.
Intermediate steps of the pipeline must be 'transforms', that is, they
must implement fit and transform methods.
The final estimator only needs to implement fit.
The transformers in the pipeline can be cached using ``memory`` argument.

The purpose of the pipeline is to assemble several steps that can be
cross-validated together while setting different parameters.
For this, it enables setting parameters of the various steps using their
names and the parameter name separated by a '__', as in the example below.
A step's estimator may be replaced entirely by setting the parameter
with its name to another estimator, or a transformer removed by setting
it to 'passthrough' or ``None``.

Read more in the :ref:`User Guide <pipeline>`.

.. versionadded:: 0.5

 Parameters

steps : list
    List of (name, transform) tuples (implementing fit/transform) that are
    chained, in the order in which they are chained, with the last object
    an estimator.

memory : str or object with the joblib.Memory interface, default=None
    Used to cache the fitted transformers of the pipeline. By default,
    no caching is performed. If a string is given, it is the path to
    the caching directory. Enabling caching triggers a clone of
    the transformers before fitting. Therefore, the transformer
    instance given to the pipeline cannot be inspected
    directly. Use the attribute ``named_steps`` or ``steps`` to
    inspect estimators within the pipeline. Caching the
    transformers is advantageous when fitting is time consuming.

verbose : bool, default=False
    If True, the time elapsed while fitting each step will be printed as it
    is completed.

Attributes
----------
named_steps : :class:`~sklearn.utils.Bunch`
    Dictionary-like object, with the following attributes.
    Read-only attribute to access any step parameter by user given name.
    Keys are step names and values are steps parameters.

See Also
--------
sklearn.pipeline.make_pipeline : Convenience function for simplified
    pipeline construction.

Examples
--------
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

 X, y = make_classification(random_state=0)

 X_train, X_test, y_train, y_test = train_test_split(X, y,
...                                                     random_state=0)
 pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

 #The pipeline can be used as any other estimator

 #and avoids leaking the test set into the train set

 pipe.fit(X_train, y_train)

 Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])

 pipe.score(X_test, y_test)

 0.88

 More:

https://pycaret.org

https://scikit-learn.org/stable

                  """)
        metadata

        st.download_button('Download Model Metadata', metadata)

    elif page == 'Model Development':
        image_pipeline = Image.open('pipeline.drawio.png')
        st.image(image_pipeline, width=1500)
        st.sidebar.image('image2.jpg')
        st.sidebar.write("""
         ## About
         Have a look to the [Notebook](https://github.com/DASA39)""")


if __name__ == '__main__':

    run()
