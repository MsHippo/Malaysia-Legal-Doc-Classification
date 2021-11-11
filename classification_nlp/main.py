#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string, re
import pandas as pd, numpy as np
import PyPDF2
import os, pickle
import pdftotext


# In[2]:


def read_pdf(file_path):
#     file_path = str(input("Enter training directory path: \n"))
    files_list = []
    # PDF documents filepath
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".pdf"):
                files_list.append(os.path.join(root, file))
    return files_list
#     for file in files_list:
#         print(file)
    


# In[3]:


def extract_pdf_content(parent_dir, directory,  file_name):
    # Path
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.mkdir(path)
#     print("Directory '% s' created" % path)
#     print("file_name '% s' created" % file_name)
#     file_name = head_list[8]
#     new_files_path = path + "/"+ file_name +'.file'
    new_files_path = path + "/"+ file_name +'.txt'
    # Save all text to a txt file.
    with open(new_files_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(pdf))
        print("Files '% s' created" % file_name)


# In[4]:


training_files_path = str(input("Enter training directory path: \n"))
# read_pdf(file_path)


# In[5]:



for file in read_pdf(training_files_path):
#     print(files_list)
    content_data= ""
#     Legal_PDF_file = open(file, 'rb')
    # Load your PDF
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f)
#     file_name = os.path.splitext(file)[0]
    head, tail = os.path.splitext(file)
#     print(file_name)
    head_list = head.split(os.sep)
    extract_pdf_content(str(r"C:/Users/heihe/Desktop/Malaysia-Legal-Doc-Classification/classification_nlp/training_data_txt/"), head_list[7], head_list[8])


# In[6]:


testing_files_path = str(input("Enter testing directory path: \n"))


# In[7]:


for file in read_pdf(testing_files_path):
#     print(files_list)
    content_data= ""
#     Legal_PDF_file = open(file, 'rb')
    # Load your PDF
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f)
#     file_name = os.path.splitext(file)[0]
    head, tail = os.path.splitext(file)
#     print(file_name)
    head_list = head.split(os.sep)
    extract_pdf_content(str(r"C:/Users/heihe/Desktop/Malaysia-Legal-Doc-Classification/classification_nlp/testing_data_txt/"),head_list[7], head_list[8])


# ## Data Preprocessing

# In[15]:


def categories_list(training_files_path):
    categories = []
    for root, dirs, files in os.walk(training_files_path):
        for i in range (len(dirs)):
            categories.append (dirs[i])
    return categories
    print (f"categories = {categories}")


# In[16]:


categories = categories_list(training_files_path)


# In[17]:


import sklearn.datasets as skd

legal_train = skd.load_files('C:/Users/heihe/Desktop/Malaysia-Legal-Doc-Classification/classification_nlp/training_data', categories= categories, encoding= 'ISO-8859-1')
legal_test = skd.load_files('C:/Users/heihe/Desktop/Malaysia-Legal-Doc-Classification/classification_nlp/testing_data',categories= categories, encoding= 'ISO-8859-1')
print ('data loaded')


# In[21]:


#categories
print(legal_train.target_names)


# ## CountVectorizer

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(legal_train.data)

# the result of shape is [n_samples, n_features]
X_train_counts.shape


# ## TF-IDF

# In[24]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# ## CountVectorizer & TF-IDF  

# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf_vect = vectorizer.fit_transform(legal_train.data)
X_train_tfidf_vect.shape


# # Multinomial Naive Bayes

# In[49]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf_vect, legal_train.target)


# In[128]:


from sklearn.pipeline import Pipeline

# stop_words : Get rid of english stop words.
#accuracy increase by setting sublinear_tf = False; 0.49 for True
text_clf = Pipeline([('vect_tfidf', TfidfVectorizer(lowercase=True, sublinear_tf=False, max_df=0.5, use_idf=True, stop_words='english')), ('clf', MultinomialNB())])
# train the model
text_clf  = text_clf.fit(legal_train.data, legal_train.target)


# In[129]:


import numpy as np
# Predict the test cases
predicted = text_clf.predict(legal_test.data)
np.mean(predicted == legal_test.target)


# In[130]:


from sklearn import metrics
from sklearn.metrics import accuracy_score

nb_classification_report_show = metrics.classification_report(legal_test.target, predicted, target_names=legal_test.target_names, output_dict=False)
print('Accuracy achieved is ' + str(np.mean(predicted == legal_test.target)))
print(nb_classification_report_show, metrics.confusion_matrix(legal_test.target, predicted))

nb_classification_report_save = metrics.classification_report(legal_test.target, predicted, target_names=legal_test.target_names, output_dict=True)

nb_classification_report_save = pd.DataFrame(nb_classification_report_save).transpose()
nb_classification_report_save.to_excel("nb_classification_report.xlsx")


# ## Visualise the confusion matrix

# In[131]:


from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

mat = confusion_matrix(legal_test.target, predicted)
nb_cfm = sns.set(font_scale=2.0)
nb_cfm = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=legal_train.target_names, yticklabels=legal_train.target_names)
nb_cfm.set_xticklabels(nb_cfm.get_xmajorticklabels(), fontsize = 14)
nb_cfm.set_yticklabels(nb_cfm.get_ymajorticklabels(), fontsize = 14)

# Set title
title_font = {'size':'21'}
nb_cfm.set_title('Naive Bayes Confusion Matrix', fontdict=title_font);
nb_cfm = plt.xlabel('True label',fontsize = 14)
nb_cfm = plt.ylabel('Predicted label', fontsize = 14)
nb_cfm = plt.gcf()

nb_cfm.set_size_inches(17, 15)
plt.savefig("nb_cfm.png", dpi=199)


# ## SVM

# In[148]:


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect_tfidf-svm', TfidfVectorizer(lowercase=True, sublinear_tf=True, max_df=0.5, stop_words='english')), 
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3,  n_iter_no_change=5, random_state=42)),])

text_clf_svm = text_clf_svm.fit(legal_train.data, legal_train.target)
predicted_svm = text_clf_svm.predict(legal_test.data)
np.mean(predicted_svm == legal_test.target)


# In[165]:


svm_classification_report_show = metrics.classification_report(legal_test.target, predicted_svm, target_names=legal_test.target_names, output_dict=False)
print('Accuracy achieved is ' + str(np.mean(predicted_svm == legal_test.target)))
print(svm_classification_report_show, metrics.confusion_matrix(legal_test.target, predicted_svm))

svm_classification_report_save = metrics.classification_report(legal_test.target, predicted_svm, target_names=legal_test.target_names, output_dict=True)

svm_classification_report_save = pd.DataFrame(svm_classification_report_save).transpose()
svm_classification_report_save.to_excel("svm_classification_report.xlsx")


# In[150]:


mat = confusion_matrix(legal_test.target, predicted_svm)
svm_cfm = sns.set(font_scale=2.0)
svm_cfm = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=legal_train.target_names, yticklabels=legal_train.target_names)
svm_cfm.set_xticklabels(svm_cfm.get_xmajorticklabels(), fontsize = 14)
svm_cfm.set_yticklabels(svm_cfm.get_ymajorticklabels(), fontsize = 14)

# Set title
title_font = {'size':'21'}
svm_cfm.set_title('SVM Confusion Matrix', fontdict=title_font);
svm_cfm = plt.xlabel('True label',fontsize = 14)
svm_cfm = plt.ylabel('Predicted label', fontsize = 14)
svm_cfm = plt.gcf()

svm_cfm.set_size_inches(17, 15)
plt.savefig("svm_cfm.png", dpi=199)


# ## Grid Search
# ### may use feature engineering instead but need use more time

# # Naive Bayes

# In[142]:


from sklearn.model_selection import GridSearchCV
parameters_nb = {
  'clf__alpha': [1e-2, 1e-3],
  'vect_tfidf__max_df': np.linspace(0.1, 1, 10),
  'vect_tfidf__binary': [True, False],
  'vect_tfidf__norm': [None, 'l1', 'l2'], 
  'vect_tfidf__use_idf': (True, False)
}

gs_clf = GridSearchCV(text_clf, parameters_nb, n_jobs=-1)
# gs_clf = GridSearchCV(text_clf, parameters_nb)
gs_clf = gs_clf.fit(legal_train.data, legal_train.target)


# In[144]:


print (gs_clf.best_score_)
print (gs_clf.best_params_)


# In[160]:


gs_predicted = gs_clf.predict(legal_test.data)

nb_classification_report_show = metrics.classification_report(legal_test.target, gs_predicted, target_names=legal_test.target_names, output_dict=False)
print(f'Accuracy achieved is {gs_clf.best_score_}')
print(nb_classification_report_show, metrics.confusion_matrix(legal_test.target, gs_predicted))

nb_classification_report_save = metrics.classification_report(legal_test.target, gs_predicted, target_names=legal_test.target_names, output_dict=True)

nb_classification_report_save = pd.DataFrame(nb_classification_report_save).transpose()

with pd.ExcelWriter('nb_classification_report.xlsx', engine='openpyxl', mode='a') as writer:  
    nb_classification_report_save.to_excel(writer, sheet_name='gs')


# In[158]:


mat = confusion_matrix(legal_test.target, gs_predicted)
nb_cfm = sns.set(font_scale=2.0)
nb_cfm = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=legal_train.target_names, yticklabels=legal_train.target_names)
nb_cfm.set_xticklabels(nb_cfm.get_xmajorticklabels(), fontsize = 14)
nb_cfm.set_yticklabels(nb_cfm.get_ymajorticklabels(), fontsize = 14)

# Set title
title_font = {'size':'21'}
nb_cfm.set_title('Naive Bayes Confusion Matrix', fontdict=title_font);
nb_cfm = plt.xlabel('True label',fontsize = 14)
nb_cfm = plt.ylabel('Predicted label', fontsize = 14)
nb_cfm = plt.gcf()

nb_cfm.set_size_inches(17, 15)
plt.savefig("nb_gs_cfm.png", dpi=199)


# # SVM

# In[153]:


parameters_svm = {'clf-svm__alpha': [1e-2, 1e-3],
  'vect_tfidf-svm__max_df': np.linspace(0.1, 1, 10),
  'vect_tfidf-svm__binary': [True, False],
  'vect_tfidf-svm__norm': [None, 'l1', 'l2'], 
  'vect_tfidf-svm__use_idf': (True, False), 
  'vect_tfidf-svm__ngram_range': [(1, 1), (1, 2)]}

gs_svm_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_svm_clf = gs_svm_clf.fit(legal_train.data, legal_train.target)
gs_svm_clf.best_score_
gs_svm_clf.best_params_


# In[154]:


print (gs_svm_clf.best_score_)
print (gs_svm_clf.best_params_)


# In[163]:


gs_predicted_svm = gs_svm_clf.predict(legal_test.data)

svm_classification_report_show = metrics.classification_report(legal_test.target, gs_predicted_svm, target_names=legal_test.target_names, output_dict=False)
print(f'Accuracy achieved is {gs_svm_clf.best_score_}')
print(svm_classification_report_show, metrics.confusion_matrix(legal_test.target, gs_predicted_svm))

svm_classification_report_save = metrics.classification_report(legal_test.target, gs_predicted_svm, target_names=legal_test.target_names, output_dict=True)

svm_classification_report_save = pd.DataFrame(svm_classification_report_save).transpose()
with pd.ExcelWriter('svm_classification_report.xlsx', engine='openpyxl', mode='a') as writer:  
    svm_classification_report_save.to_excel(writer, sheet_name='gs')


# In[164]:


mat = confusion_matrix(legal_test.target, gs_predicted_svm)
svm_cfm = sns.set(font_scale=2.0)
svm_cfm = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=legal_train.target_names, yticklabels=legal_train.target_names)
svm_cfm.set_xticklabels(svm_cfm.get_xmajorticklabels(), fontsize = 14)
svm_cfm.set_yticklabels(svm_cfm.get_ymajorticklabels(), fontsize = 14)

# Set title
title_font = {'size':'21'}
svm_cfm.set_title('SVM Confusion Matrix', fontdict=title_font);
svm_cfm = plt.xlabel('True label',fontsize = 14)
svm_cfm = plt.ylabel('Predicted label', fontsize = 14)
svm_cfm = plt.gcf()

svm_cfm.set_size_inches(17, 15)
plt.savefig("svm_gs_cfm.png", dpi=199)

