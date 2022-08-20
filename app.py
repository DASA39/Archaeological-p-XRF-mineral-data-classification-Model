from ssl import Options
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
         If you're seeing this, we would love your contribution! If you find bugs, please reach out or create an issue on our 
         [GitHub](https://github.com/DASA39) repository. If you find that this interface doesn't do what you need it to, you can create an feature request 
         at our repository or better yet, contribute a pull request of your own. You can reach out to the team on LinkedIn.
    
         More documentation and contribution details are at our [GitHub Repository](https://github.com/DASA39).
        
         This app is the result of a PhD ongoing project titled: Exploring complexity through amber and variscite.
         Computational archaeology and geoarchaeological data in the Late Prehistory of Iberia.
         A project develped by:
         
        -Daniel Sánchez-Gómez
        """
    })

timestr = time.strftime("%Y%m%d-%H%M")
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

        st.title("Archaeological stone classification project")
        file_upload = st.file_uploader(
            "Upload csv file for predictions", type=["csv"])

        if file_upload is not None:

            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)
            csv_downloader(predictions)
        st.sidebar.write("""
         ## About
          This is the core of the application. A Machine Learning model has been developed and trained to predict the mineral class to which a sample obtained from XRF belongs.
          Once a csv file has been upload, the model will predict and display the results in a new column called Label as well
          as the probability with which the algorithm scores its prediction.
          """)

    elif page == 'Project':
        st.header("Exploring complexity through amber and variscite. Computational archaeology and geoarchaeological data in the Late Prehistory of the Iberian peninsula. (4th-to 2nd millennia BC)")
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
data of thousands of items of personal adornment made out of amber and variscite from 
more than 900 archaeological sites on the Iberian Peninsula through different projects, 
which represents a first-rate experimental data set for the study of these subjects.
Despite the existence of such relevant data sets, to date, there are both methodological
and theoretical challenges in extracting knowledge from this type of resources due to the
lack of comprehensive studies with a data-centred approach. Regarding the study of raw
materials like amber or variscite minerals, it is still necessary to make exhaustive
inventories of Iberian sources, mineralogical characterisation of items, and systematisation 
of scattered and unpublished data among other urgent tasks that will improve or 
reconsider provenance models used to explain the socio-economic dynamics in late prehistory.
Through the use of different techniques of Computational archaeology such as Data
mining and Machine Learning, the aim of this doctoral program is to explore a data driven
approach to solve some of the main methodological challenges in the study of the
socio-economic complexity in the late prehistory of the Iberian Peninsula and develop an
Open Access approach for the publication of results in accordance with the necessity of
digitalization of humanities.

         """)
        st.sidebar.write("""
         ## About

### Summary:

Exploring complexity through amber and variscite. Computational archaeology and
geoarchaeological data in the Late Prehistory of the Iberian peninsula. 
(4th-to 2nd millennia BC) is an ongoing PhD project developed in the [University of Seville](http://institucional.us.es/doctorhistoria/)


The aim of this doctoral program is to explore a data driven
approach to solve some of the main methodological challenges in the study of the
socio-economic complexity in the late prehistory of the Iberian Peninsula and develop an
Open Access approach for the publication of results in accordance with the necessity of
digitalization of humanities.""")
        abstract

    elif page == 'Model Metadata':
        image_pipeline = Image.open('pipeline.drawio.png')
        st.image(image_pipeline,use_column_width='auto')
        st.sidebar.image('image.jpg')
        st.sidebar.write("""
         ## About""")
        metadata = (
            """Pipeline(steps=[('dtypes',
                 DataTypes_Auto_infer(ml_usecase='classification',
                                      target='target')),
                ('imputer',
                 Simple_Imputer(categorical_strategy='not_available',
                                fill_value_categorical=None,
                                fill_value_numerical=None,
                                numeric_strategy='mean',
                                target_variable=None)),
                ('new_levels1',
                 New_Catagorical_Levels_in_TestData(replacement_strategy='least '
                                                                         'frequent',
                                                    targ...
                ('clean_names', Clean_Colum_Names()),
                ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                ('dfs', 'passthrough'), ('pca', 'passthrough'),
                ['trained_model',
                 ExtraTreesClassifier(class_weight={}, criterion='entropy',
                                      max_depth=5, max_features=1.0,
                                      min_impurity_decrease=0.0002,
                                      min_samples_leaf=5, min_samples_split=10,
                                      n_estimators=150, n_jobs=-1,
                                      random_state=123)]])        

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
        st.sidebar.image('image2.jpg')
        st.sidebar.write("""
         ## About
         Aqui va el enlace directo al notebook del modelo
         [GitHub](https://github.com/DASA39)""")

    # if file_upload is not None:
       #data = pd.read_csv(file_upload)
       #predictions = predict_model(estimator=model,data=data)
       # st.write(predictions)
       # csv_downloader(predictions)


# Función de Interfaz de Usuario
# def run_UI():


if __name__ == '__main__':

    run()
