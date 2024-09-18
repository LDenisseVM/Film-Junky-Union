#!/usr/bin/env python
# coding: utf-8

# ¡Hola Denisse! Como te va?
# 
# Mi nombre es Facundo Lozano! Un gusto conocerte, seré tu revisor en este proyecto.
# 
# A continuación un poco sobre la modalidad de revisión que usaremos:
# 
# Cuando enccuentro un error por primera vez, simplemente lo señalaré, te dejaré encontrarlo y arreglarlo tú cuenta. Además, a lo largo del texto iré haciendo algunas observaciones sobre mejora en tu código y también haré comentarios sobre tus percepciones sobre el tema. Pero si aún no puedes realizar esta tarea, te daré una pista más precisa en la próxima iteración y también algunos ejemplos prácticos. Estaré abierto a comentarios y discusiones sobre el tema.
# 
# Encontrará mis comentarios a continuación: **no los mueva, modifique ni elimine**.
# 
# Puedes encontrar mis comentarios en cuadros verdes, amarillos o rojos como este:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Exito. Todo se ha hecho de forma exitosa.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Observación. Algunas recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Necesita arreglos. Este apartado necesita algunas correcciones. El trabajo no puede ser aceptado con comentarios rojos. 
# </div>
# 
# Puedes responder utilizando esto:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div>
# 

# # Descripcipción del proyecto

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 2) </b> <a class="tocSkip"></a>
# 
# Felicitaciones Denisse, has implementado los cambios que te he indicado, bien hecho! Hemos alcanzado los objetivos del proyecto por lo que está en condiciones de ser aprobado!
#     
# Felicitaciones y éxitos en tu camino dentro del mundo de los datos!
#     
# Saludos!

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 1) </b> <a class="tocSkip"></a>
# 
# Denisse, siempre me tomo este tiempo al inicio de tu proyecto para comentarte mis apreciaciones generales de esta iteración de tu entrega.
# 
# Me gusta comenzar dando la bienvenida al mundo de los datos a los estudiantes, te deseo lo mejor y espero que consigas lograr tus objetivos. Personalmente me gusta brindar el siguiente consejo, "Está bien equivocarse, es normal y es lo mejor que te puede pasar. Aprendemos de los errores y eso te hará mejor programadora ya que podrás descubrir cosas a medida que avances y son estas cosas las que te darán esa experiencia para ser una gran Data Scientist"
#     
# Ahora si yendo a esta notebook. Quería felicitarte Denisse porque has logrado resolver todos los pasos implementando grandes lógicas, se ha notado tu manejo sobre python y las herramientas ML utilizadas. Muy bien hecho! Por otro lado hemos tenido un par de detalles que debemos corregir pero que estoy seguro que te demorara unos minutos, te he dejado los comentarios con el contexto necesario para que puedas resolverlo.
#     
# Éxitos y espero con ansias a nuestra próxima iteración!
#     
# Saludos!

# Film Junky Union, una nueva comunidad vanguardista para los aficionados de las películas clásicas, está desarrollando un sistema para filtrar y categorizar reseñas de películas. Tu objetivo es entrenar un modelo para detectar las críticas negativas de forma automática. Para lograrlo, utilizarás un conjunto de datos de reseñas de películas de IMDB con leyendas de polaridad para construir un modelo para clasificar las reseñas positivas y negativas. Este deberá alcanzar un valor F1 de al menos 0.85.

# ## Inicialización

# In[1]:


import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tqdm.auto import tqdm

import re


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
# la siguiente línea proporciona gráficos de mejor calidad en pantallas HiDPI
# %config InlineBackend.figure_format = 'retina'

plt.style.use('seaborn')


# In[3]:


# esto es para usar progress_apply, puedes leer más en https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()


# ## Cargar datos

# In[4]:


df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})


# ## EDA

# In[5]:


df_reviews.head()


# In[6]:


df_reviews.info()


# In[7]:


df_reviews.describe()


# In[8]:


df_reviews= df_reviews.drop_duplicates()


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
#     
# Excelente Denisse, hasta el momento hemos cargado correctamente los datos separandolos de las importaciones para mitigar posibles errores, a la vez has implementado métodos para comprender la composición de nuestros datos. Bien hecho!

# <div class="alert alert-block alert-warning">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
#     
# Un agregado que podríamos implementar es la observación de valos ausentes o incluso duplicados.

# Veamos el número de películas y reseñas a lo largo de los años.

# In[9]:


fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Número de películas a lo largo de los años')

ax = axs[1]


dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Número de reseñas a lo largo de los años')

fig.tight_layout()


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Unas observaciones que nos permitirán comprender el trabajo que debemos realizar. Excelente

# Veamos la distribución del número de reseñas por película con el conteo exacto y KDE (solo para saber cómo puede diferir del conteo exacto)

# In[10]:


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Gráfico de barras de #Reseñas por película')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('Gráfico KDE de #Reseñas por película')

fig.tight_layout()


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Profundizamos sobre la cantidad de reseñas, bien hecho!

# In[11]:


df_reviews['pos'].value_counts()


# In[12]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de entrenamiento: distribución de puntuaciones')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de prueba: distribución de puntuaciones')

fig.tight_layout()


# Distribución de reseñas negativas y positivas a lo largo de los años para dos partes del conjunto de datos

# In[13]:


fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de entrenamiento: distribución de diferentes polaridades por película')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de prueba: número de reseñas de diferentes polaridades por año')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de prueba: distribución de diferentes polaridades por película')

fig.tight_layout()


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
#     
# Nuevamente excelente! Continuamos el proceso de comprensión de los datos y del problema.

# In[14]:


df_reviews['pos'].value_counts()


# Vemos que hubo un aumento en el número de peliculas, y con ello de reseñas a lo largo de los años, sin embargo en la última década, hubo una diminución
# En cuanto a la desequilibrio de clases, vemos que está muy parejo, por lo que eso no será un problema para nuestro modelo 

# ## Procedimiento de evaluación

# Composición de una rutina de evaluación que se pueda usar para todos los modelos en este proyecto

# In[15]:


import sklearn.metrics as metrics
def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'Curva ROC')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return


# ## Normalización

# Suponemos que todos los modelos a continuación aceptan textos en minúsculas y sin dígitos, signos de puntuación, etc.

# In[16]:


def prep_text(text):
    text = text.lower()
    pattern = r"[^a-z ]"
    text = re.sub(pattern, " ", text)
    text = text.split()
    text = " ".join(text)
    return text

df_reviews['review_norm'] = df_reviews['review'].apply(prep_text)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
#     
# Excelente creación de la función para la normalización de nuestro corpus, muy bien hecho Denisse!

# ## División entrenamiento / prueba

# Por fortuna, todo el conjunto de datos ya está dividido en partes de entrenamiento/prueba; 'ds_part' es el indicador correspondiente.

# In[17]:


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Partimos los datos en los conjuntos de entrenamiento y testeo. Bien! Sigamos!

# ## Trabajar con modelos

# ### Modelo 0 - Constante

# In[18]:


from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import spacy
import nltk


# In[19]:


df_reviews_train.dropna(subset=['review_norm'], inplace=True)
df_reviews_test.dropna(subset=['review_norm'], inplace=True)

lemmas_train= df_reviews_train['review_norm'].copy()
lemmas_test= df_reviews_test['review_norm'].copy()


dummy_classifier = DummyClassifier(strategy="constant", constant=1)

# Entrenar el clasificador Dummy
dummy_classifier.fit(lemmas_train, train_target)

# Realizar predicciones en el conjunto de prueba
predictions = dummy_classifier.predict(lemmas_test)

# Calcular la precisión del clasificador Dummy
accuracy= dummy_classifier.score(predictions, test_target)
print("Accuracy del clasificador Dummy:", accuracy)


# In[20]:


evaluate_model(dummy_classifier, lemmas_train, train_target, lemmas_test, test_target)


# <div class="alert alert-block alert-warning">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Aquí hemos implementado una lógica correcta, sin embargo nos hemos confundido en 2 detalles, por un lado un modelo constante nos implica utilizar el parametro **constant** sobre Strategy, en agregado deberíamos agregar el parametro **constant=1**

# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 2)</b> <a class="tocSkip"></a>
# 
# Ahora si excelente Denisse, lo hemos implementado tal como debíamos!

# ### Modelo 1 - NLTK, TF-IDF y LR

# TF-IDF

# In[21]:


import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[22]:


lemmatizer = WordNetLemmatizer()

corpus_train = []
for review in df_reviews_train['review_norm']:
    tokens_t = word_tokenize(review)
    lemmas_train = [lemmatizer.lemmatize(token) for token in tokens_t]
    corpus_train.append(" ".join(lemmas_train))


# In[23]:


corpus_test = []
for review in df_reviews_test['review_norm']:
    tokens_t = word_tokenize(review)
    lemmas_test = [lemmatizer.lemmatize(token) for token in tokens_t]
    corpus_test.append(" ".join(lemmas_test))


# In[24]:


nltk.download('stopwords')

stop_words= set(stopwords.words('english'))

count_tf_idf_1= TfidfVectorizer(stop_words= stop_words)
tf_idf_1= count_tf_idf_1.fit_transform(corpus_train)

train_features_1=tf_idf_1
test_features_1= count_tf_idf_1.transform(corpus_test)

model_1= LogisticRegression(solver='liblinear')
model_1.fit(train_features_1, train_target)
pred_test= model_1.predict(test_features_1)


# In[25]:


evaluate_model(model_1, train_features_1, train_target, test_features_1, test_target)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Una implementación perfecta Denisse, desde la utilziación de las stopwwords provistas por nltk hasta la creación del vectorizador, la lematización y el testeo final utilizando la función evaluate_model. Sigamos

# ### Modelo 3 - spaCy, TF-IDF y LR

# In[26]:


import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[27]:


def text_preprocessing_3(text):
    
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
    return ' '.join(tokens)


# In[28]:


corpus_train_3= df_reviews_train['review_norm'].apply(text_preprocessing_3)
corpus_test_3= df_reviews_test['review_norm'].apply(text_preprocessing_3)

count_tf_idf_3= TfidfVectorizer()
tf_idf_3= count_tf_idf_3.fit_transform(corpus_train_3)

train_features_3=tf_idf_3
test_features_3= count_tf_idf_3.transform(corpus_test_3)

model_3= LogisticRegression(solver='liblinear')
model_3.fit(train_features_3, train_target)
pred_test= model_3.predict(test_features_3)


# In[29]:


evaluate_model(model_3, train_features_3, train_target, test_features_3, test_target)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Nuevamente perfecto, en este caso utilizando spacy para la tokenización del corpus y la obtención de lemma de cada token. Excelente Denisse!

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Un detalle importante que nos hemos olvidado aquí Denisse es que debemos implementar las stopwords pero mediante spacy, en este caso hemos utilizado nltk. Trata de implementar el filtrado con stopwords de spacy.

# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 2)</b> <a class="tocSkip"></a>
# 
# Bien hecho Denisse, lo implementaste tal como debíamos!

# ### Modelo 4 - spaCy, TF-IDF y LGBMClassifier

# In[30]:


from lightgbm import LGBMClassifier as lgb


# In[31]:


train_features_4 = train_features_3.copy()
test_features_4 = test_features_3.copy()


# In[32]:


params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.1
}

# Crear el modelo LightGBM
model_4 = lgb(**params)

# Entrenar el modelo LightGBM
model_4.fit(train_features_4, train_target, eval_set=[(test_features_4, test_target)], early_stopping_rounds=50, verbose=100)


# In[33]:


evaluate_model(model_4, train_features_4, train_target, test_features_4, test_target)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Una aplicación correcta Denisse, en este caso optimizando código y utilizando los datos obtenidos de spacy en el modelo anterior. Bien hecho!

# ###  Modelo 9 - BERT

# In[ ]:


import torch
import transformers


# In[ ]:


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')


# In[ ]:


def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, force_device=None, disable_progress_bar=False):
    
    ids_list = []
    attention_mask_list = []

    # texto al id de relleno de tokens junto con sus máscaras de atención 
       
    # <escribe tu código aquí para crear ids_list y attention_mask_list>
    
    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    if not disable_progress_bar:
        print(f'Uso del dispositivo {device}.')
    
    # obtener insertados en lotes
    
    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=disable_progress_bar):
            
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        # <escribe tu código aquí para crear attention_mask_batch
            
        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())
        
    return np.concatenate(embeddings)


# In[ ]:


# ¡Atención! La ejecución de BERT para miles de textos puede llevar mucho tiempo en la CPU, al menos varias horas
train_features_9 = BERT_text_to_embeddings(df_reviews_train['review_norm'], force_device='cuda')


# In[ ]:


print(df_reviews_train['review_norm'].shape)
print(train_features_9.shape)
print(train_target.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# si ya obtuviste los insertados, te recomendamos guardarlos para tenerlos listos si
# np.savez_compressed('features_9.npz', train_features_9=train_features_9, test_features_9=test_features_9)

# y cargar...
# with np.load('features_9.npz') as data:
#     train_features_9 = data['train_features_9']
#     test_features_9 = data['test_features_9']


# In[ ]:





# In[ ]:





# In[ ]:





# ## Mis reseñas

# In[38]:


my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])


my_reviews['review_norm'] = my_reviews['review'].apply(prep_text)

my_reviews


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Excelentes reviews!

# ### Modelo 1

# In[39]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_1.predict_proba(count_tf_idf_1.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 3

# In[40]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_3.predict_proba(count_tf_idf_3.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Modelo 4

# In[41]:


texts = my_reviews['review_norm']

tfidf_vectorizer_4 = count_tf_idf_3
my_reviews_pred_prob = model_4.predict_proba(tfidf_vectorizer_4.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Implementaciones y resultados perfectos Denisse! Bien hecho!

# ### Modelo 9

# In[ ]:


texts = my_reviews['review_norm']

my_reviews_features_9 = BERT_text_to_embeddings(texts, disable_progress_bar=True)

my_reviews_pred_prob = model_9.predict_proba(my_reviews_features_9)[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ## Conclusiones

# Vemos que cada modelo calcula distintas probabilidades para las reseñas de prueba, sin embargo austando el umbral, son bastante exactas, aún con sus diferencias, lo que va de acuerdo con las pruebas que hicimos previamente con el dataset de train y test, segun nuestra evaluación previa
# 
# Probamos distintos modelos, con distintas técnicas de preprocesamiento de datos y obtuvimos resultados bastante sobresalientes, tanto en train como en test, y con las reseñas extras de prueba

# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Una conclusión que demuestra la comprensión de lo hecho a lo largo del trabajo, bien hecho Denisse!

# # Lista de comprobación

# - [x]  Abriste el notebook
# - [x]  Cargaste y preprocesaste los datos de texto para su vectorización
# - [x]  Transformaste los datos de texto en vectores
# - [x]  Entrenaste y probaste los modelos
# - [x]  Se alcanzó el umbral de la métrica
# - [x]  Colocaste todas las celdas de código en el orden de su ejecución
# - [x]  Puedes ejecutar sin errores todas las celdas de código 
# - [x]  Hay conclusiones 

# In[ ]:




