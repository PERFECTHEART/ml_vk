#!/usr/bin/env python
# coding: utf-8

# ### Машинное Обучение
# 
# ## Домашнее задание №2 - Дерево решений

# **Общая информация**
# 
# **Срок сдачи:** 1 февраля 2023, 08:30   
# **Штраф за опоздание:** -2 балла за каждые 2 дня опоздания
# 
# Решений залить в свой github репозиторий.
# 
# Используйте данный Ipython Notebook при оформлении домашнего задания.

# ##  Реализуем дерево решений (3 балла)

# Допишите недостающие части дерева решений. Ваша реализация дерева должна работать по точности не хуже DecisionTreeClassifier из sklearn.
# Внимание: если Вас не устраивает предложенная структура хранения дерева, Вы без потери баллов можете сделать свой класс MyDecisionTreeClassifier, в котором сами полностью воспроизведете алгоритм дерева решений. Обязательно в нем иметь только функции fit, predict . (Но название класса не менять)

# In[1]:


import numpy as np
import pandas as pd
#import scipy.stats

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


class MyDecisionTreeClassifier:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1
    
    def __init__(self, min_samples_split=2, max_depth=5, criterion='gini'):
        """
        criterion -- критерий расщепления. необходимо релизовать три:
        Ошибка классификации, Индекс Джини, Энтропийный критерий
        max_depth -- максимальная глубина дерева
        min_samples_split -- минимальное число объектов в листе, чтобы сделать новый сплит
        """
        self.min_samples_split, self.max_depth, self.num_class, self.criterion, self.__y_criterion,         self.feature_importances_ = min_samples_split, max_depth, None, criterion, 0, None
        
        def __misclass(y):
            if y.size <= 1:
                return 1.0
            else:
                return 1.0 - max(np.bincount(y))/y.size

        def __gini(y):
            return 1.0 - sum(np.square(np.bincount(y)/y.size))

        def __entropy(y):
            if y.size <= 1:
                return 0
            ps = np.bincount(y) / y.size
            return -sum([p * np.log2(p) for p in ps if p > 0])
       
        # Структура, которая описывает дерево
        # Представляет словарь, где для  node_id (айдишник узла дерева) храним
        # (тип_узла, айдишник признака сплита, порог сплита) если тип NON_LEAF_TYPE
        # (тип_узла, предсказание класса, вероятность класса) если тип LEAF_TYPE
        # Подразумевается, что у каждого node_id в дереве слева 
        # узел с айди 2 * node_id + 1, а справа 2 * node_id + 2
        self.tree = dict()
        if criterion == 'gini':
            self.calc_criterion = __gini
        elif self.criterion == 'misclass':
            self.calc_criterion = __misclass
        elif self.criterion == 'entropy':
            self.calc_criterion = __entropy
        else:
            raise NameError("No such criterion")

    def __div_samples(self, x, y, feature_id, threshold):
        """
        Разделяет объекты на 2 множества
        x -- матрица объектов
        y -- вектор ответов
        feature_id -- айдишник признака, по которому делаем сплит
        threshold -- порог, по которому делаем сплит
        """
        left_mask = x[:, feature_id] > threshold
        right_mask = ~left_mask
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]
    
    def __calc_Q(self, y, i):
        y_l, y_r = y[0:i], y[i:y.shape[0]]
        #Q = self.__y_criterion - (self.calc_criterion(y_l)*y_l.shape[0] + self.calc_criterion(y_r)*y_r.shape[0])/y.shape[0]
        return self.__y_criterion-(self.calc_criterion(y_l)*y_l.shape[0] + self.calc_criterion(y_r)*y_r.shape[0])/y.shape[0] #Q
  
    def __find_threshold(self, x, y):
        """
        Находим оптимальный признак и порог для сплита
        Здесь используемые разные impurity в зависимости от self.criterion
        """
        opt_feature, opt_split, max_Q, self.__y_criterion = 0, x[0, 0], self.__calc_Q(y, 0), self.calc_criterion(y)
        #sum_y = np.sum(y)
        #sum_square_y = np.sum(np.square(y))
        for feature_idx in range(x.shape[1]):
            order = np.argsort(x[:, feature_idx])
            if np.unique(x[:, feature_idx]).size == 1: # Проверяем на уникальность
                continue
            srt_x, srt_y, y_prev_idx, same_val = x[order, :], y[order], 0, False
            for i in range(1, x.shape[0] - 1):
                if srt_y[i] == srt_y[y_prev_idx]:
                    same_val = True
                    continue
                y_prev_idx, same_val, new_Q = i, False, self.__calc_Q(srt_y, i)
                if new_Q > max_Q:
                    max_Q, opt_feature, opt_order = new_Q, feature_idx, order
                    opt_split = (srt_x[i - 1, opt_feature] + srt_x[i, opt_feature])/2
        return opt_feature, opt_split, max_Q
    '''
    def __leaf_class_predict(self, y):
        return np.argmax(np.bincount(y))
    '''
    def __fit_node(self, x, y, node_id, depth):
        """
        Делаем новый узел в дереве
        Решаем, терминальный он или нет
        Если нет, то строим левый узел  с айди 2 * node_id + 1
        И правый узел с  айди 2 * node_id + 2
        """
        if (depth >= self.max_depth) or (y.shape[0] < self.min_samples_split):
            pred = np.argmax(np.bincount(y))
            prob = np.sum(y == pred)/y.shape[0]
            self.tree[node_id] = (self.LEAF_TYPE, pred, prob) #new_node
        else:
            opt_feature, opt_split, max_Q = self.__find_threshold(x, y)
            reord_feat = x[:, opt_feature]
            l_idx = reord_feat < opt_split
            r_idx = ~l_idx # reord_feat >= opt_split
            if y[l_idx].shape[0] < self.min_samples_split or y[r_idx].shape[0] < self.min_samples_split:
                pred = np.argmax(np.bincount(y))
                prob = np.sum(y == pred)/y.shape[0]
                self.tree[node_id] = (self.LEAF_TYPE, pred, prob)
            else:
                self.tree[node_id] = (self.NON_LEAF_TYPE, opt_feature, opt_split)
                self.__fit_node(x[l_idx], y[l_idx], 2 * node_id + 1, depth + 1)
                self.__fit_node(x[r_idx], y[r_idx], 2 * node_id + 2, depth + 1)
                self.feature_importances_[opt_feature] += max_Q
                
    def fit(self, x, y):
        """
        Рекурсивно строим дерево решений
        Начинаем с корня node_id 0
        """
        self.feature_importances_ = dict.fromkeys(range(x.shape[1]), 0)
        self.num_class = len(set(y))
        
        self.__fit_node(x, y, 0, 0) 
        
    def __predict_class(self, x, node_id):
        """
        Рекурсивно обходим дерево по всем узлам,
        пока не дойдем до терминального
        """
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node
            if x[feature_id] < threshold:
                return self.__predict_class(x, 2 * node_id + 1)
            else:
                return self.__predict_class(x, 2 * node_id + 2)
        else:
            return node[1]
        
    def predict(self, X):
        """
        Вызывает predict для всех объектов из матрицы X
        """
        return np.array([self.__predict_class(x, 0) for x in X])
    
    def fit_predict(self, x_train, y_train, predicted_x):
        self.fit(x_train, y_train)
        return self.predict(predicted_x)
    
    def get_feature_importance(self):
        """
        Возвращает важность признаков
        """
        #importances = np.zeros(len(self.feature_importances_))
        #print(self.feature_importances_.items())
        #importances = [ i in self.feature_importances(,:)]
        #for k, i in self.feature_importances_.items():
        #    importances[k] = i
        importances = np.array([x[1] for x in self.feature_importances_.items()])
        return importances/np.sum(importances)


# In[3]:


my_clf = MyDecisionTreeClassifier(min_samples_split=2)
clf = DecisionTreeClassifier(min_samples_split=2)


# In[4]:


wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, stratify=wine.target)


# In[5]:


clf.fit(X_train, y_train)
print("clf: ",accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))

my_clf.fit(X_train, y_train)
print("my_clf: ",accuracy_score(y_pred=my_clf.predict(X_test), y_true=y_test))


# In[ ]:





# Совет: Проверьте, что ваша реализация корректно работает с признаками в которых встречаются повторы. 
# И подумайте, какие еще граничные случаи могут быть.
# Например, проверьте, что на таком примере ваша модель корректно работает:

# In[6]:


X = np.array([[1] * 10, [0, 1, 2, 5, 6, 3, 4, 7, 8, 9]]).T
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
for depth in range(1, 5):
    my_clf = MyDecisionTreeClassifier(max_depth=depth)
    my_clf.fit(X, y)
    print("DEPTH:", depth, "\tMyTree:", my_clf.tree, my_clf.predict(X))
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X, y)
    print("\t\tTree:", clf.tree_.threshold, clf.predict(X))


# ### Придумайте интересные примеры для отладки дерева решений (доп. задание)
# Это необязательный пункт. За него можно получить 1 доп балл. 
# Можете придумать примеры для отладки дерева, похожие на пример выше. 
# 
# Они должны быть не сложные, но в то же время информативные, чтобы можно было понять, что реализация содержит ошибки.
# Вместе с примером нужно указать ожидаемый выход модели. 

# In[7]:


# Примеры


# ## Ускоряем дерево решений (2 балла)
# Добиться скорости работы на fit не медленнее чем в 10 раз sklearn на данных wine. 
# Для этого используем numpy.

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, stratify=wine.target)
my_clf = MyDecisionTreeClassifier(min_samples_split=2)
clf = DecisionTreeClassifier(min_samples_split=2)
get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[9]:


get_ipython().run_line_magic('time', 'my_clf.fit(X_train, y_train)')


# ## Боевое применение (3 балла)
# 
# На практике Вы познакомились с датасетом Speed Dating Data. В нем каждая пара в быстрых свиданиях характеризуется определенным набором признаков. Задача -- предсказать, произойдет ли матч пары (колонка match). 
# 
# Данные и описания колонок во вложениях.
# 
# Пример работы с датасетом можете найти в практике пункт 2
# https://github.com/VVVikulin/ml1.sphere/blob/master/2019-09/lecture_06/pract-trees.ipynb
# 
# Либо воспользоваться функцией:

# In[10]:


def preprocess_spd_data(df):
    df = df.iloc[:, :97]
    
    to_drop = [
        'id', 'idg', 'condtn', 'round', 'position', 'positin1', 'order', 'partner', 
        'age_o', 'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha',
        'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o','met_o',
        'field', 'undergra', 'from', 'zipcode', 'income', 'career', 'sports', 'tvsports', 'exercise', 
        'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 
        'concerts', 'music', 'shopping', 'yoga', 'expnum',
        'mn_sat', 'tuition'
    ]

    df = df.drop(to_drop, axis=1)
    df = df.dropna(subset=['age', 'imprelig', 'imprace', 'date'])

    df.loc[:, 'field_cd'] = df.loc[:, 'field_cd'].fillna(19)
    df.loc[:, 'career_c'] = df.loc[:, 'career_c'].fillna(18)
    
    # attr1 processing
    df.loc[:, 'temp_totalsum'] = df.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 
                                            'amb1_1', 'shar1_1']].sum(axis=1)
    df.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']] =    (df.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].T / 
     df.loc[:, 'temp_totalsum'].T).T * 100
    
    # attr2 processing
    df.loc[:, 'temp_totalsum'] = df.loc[:, ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 
                                            'amb2_1', 'shar2_1']].sum(axis=1)
    df.loc[:, ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']] =    (df.loc[:, ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']].T / 
     df.loc[:, 'temp_totalsum'].T).T * 100
    df = df.drop(['temp_totalsum'], axis=1)
    
    for i in [4, 5]:
        feat = ['attr{}_1'.format(i), 'sinc{}_1'.format(i), 
                'intel{}_1'.format(i), 'fun{}_1'.format(i), 
                'amb{}_1'.format(i), 'shar{}_1'.format(i)]

        if i != 4:
            feat.remove('shar{}_1'.format(i))
    
        df = df.drop(feat, axis=1)
    
    df = df.drop(['wave'], axis=1)
    df = df.dropna()
    return df


# Скачайте датасет, обработайте данные, как показано на семинаре или своим собственным способом. Обучите дерево классифкации. В качестве таргета возьмите колонку 'match'. Постарайтесь хорошо обработать признаки, чтобы выбить максимальную точность. Если точность будет близка к случайному гаданию, задание не будет защитано. В качестве метрики можно взять roc-auc. 
# 

# In[11]:


data = pd.read_csv('./data/speed-dating-experiment/Speed_Dating_Data.csv', encoding='latin1')


# In[12]:


data.shape
data.drop(['id'], axis=1)
data.drop(['idg'], axis=1)
data.drop_duplicates(subset=['iid']).gender.value_counts()
data = data.drop(['round', 'position', 'order','field',
                  'partner', 'age_o', 'race_o', 'pf_o_att', 
                  'pf_o_sin', 'pf_o_int',
                  'pf_o_fun', 'pf_o_amb', 'pf_o_sha',
                  'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o',
                  'amb_o', 'shar_o', 'like_o', 'prob_o','met_o'], 
                 axis=1)
data.drop_duplicates(subset=['iid']).age.hist(bins=20)
data = data.dropna(subset=['age'])
data.loc[:, 'field_cd'] = data.loc[:, 'field_cd'].fillna(19)
def mean_target_encoding(data, target, column):
    mean_enc = data.groupby(column)[target].mean()
    data[column+'_m_enc'] = data[column].map(mean_enc)
    return data
data = mean_target_encoding(data, 'match', 'field_cd')


# In[13]:


data.field_cd_m_enc
data = data.drop(['field_cd'], axis=1)
data = data.drop(['undergra'], axis=1)
data.loc[:, 'mn_sat'] = data.loc[:, 'mn_sat'].str.replace(',', '').astype(float)
data.drop_duplicates('iid').mn_sat.hist()
data['mn_sat'] = data['mn_sat'].fillna(0)
data.drop_duplicates('iid').mn_sat.hist()
data.loc[:, 'tuition'] = data.loc[:, 'tuition'].str.replace(',', '').astype(float)
data.drop_duplicates('iid').tuition.hist()
data['tuition'] = data['tuition'].fillna(-10000)
data = mean_target_encoding(data, 'match', 'race')
data = data.drop(['race'], axis=1)
data.drop_duplicates('iid').imprace.isnull().sum()


# In[14]:


data.drop_duplicates('iid').imprelig.isnull().sum()


# In[15]:


data = data.dropna(subset=['imprelig', 'imprace'])
data = data.drop(['from', 'zipcode'], axis=1)
data.loc[:, 'income'] = data.loc[:, 'income'].str.replace(',', '').astype(float)
data = data.drop(['income'], axis=1)
data = mean_target_encoding(data, 'match', 'goal')
data = mean_target_encoding(data, 'match', 'date')
data = mean_target_encoding(data, 'match', 'go_out')
data = data.drop(['goal'], axis=1)
data = data.drop(['date'], axis=1)
data = data.drop(['go_out'], axis=1)
data = mean_target_encoding(data, 'match', 'career_c')
data = data.drop(['career_c', 'career'], axis=1)
data = data.drop(['sports','tvsports','exercise','dining','museums','art','hiking','gaming',
       'clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga'], axis=1)
data = data.drop(['expnum'], axis=1)
feat = ['iid', 'wave', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
temp = data.drop_duplicates(subset=['iid', 'wave']).loc[:, feat]
temp.loc[:, 'totalsum'] = temp.iloc[:, 2:].sum(axis=1)
idx = ((temp.wave < 6) | (temp.wave > 9)) & (temp.totalsum < 99)
temp.loc[idx, ]
idx = ((temp.wave >= 6) & (temp.wave <= 9))
temp.loc[idx, ]
data.loc[:, 'temp_totalsum'] = data.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].sum(axis=1)
data.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']] = (data.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].T/data.loc[:, 'temp_totalsum'].T).T * 100
feat = ['iid', 'wave', 'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']
temp = data.drop_duplicates(subset=['iid', 'wave']).loc[:, feat]
temp.loc[:, 'totalsum'] = temp.iloc[:, 2:].sum(axis=1)
idx = ((temp.wave < 6) | (temp.wave > 9)) & (temp.totalsum < 90) & (temp.totalsum != 0)
temp.loc[idx, ]
idx = ((temp.wave >= 6) & (temp.wave <= 9))
temp.loc[idx, ]
data.loc[:, 'temp_totalsum'] = data.loc[:, ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']].sum(axis=1)
data.loc[:, ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']] = (data.loc[:, ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']].T/data.loc[:, 'temp_totalsum'].T).T * 100
data = data.drop(['temp_totalsum'], axis=1)
for i in [4, 5]:
    feat = ['attr{}_1'.format(i), 'sinc{}_1'.format(i), 
            'intel{}_1'.format(i), 'fun{}_1'.format(i), 
            'amb{}_1'.format(i), 'shar{}_1'.format(i)]
    
    if i != 4:
        feat.remove('shar{}_1'.format(i))
    
    data = data.drop(feat, axis=1)
data = data.drop('positin1', axis=1)
data = data.dropna(subset=['career_c_m_enc'])
data = data.drop(['id'], axis=1)
features = list(data.columns)
bad_features = list(filter(lambda x: x[-1].isnumeric() and int(x[-1]) > 1, features))
data = data.drop(bad_features, axis=1)
bad_features = list(filter(lambda x: x[-2:] == '_s', features))
#bad_features ['attr1_s','sinc1_s','intel1_s','fun1_s','amb1_s','shar1_s','attr3_s','sinc3_s','intel3_s','fun3_s','amb3_s']
data = data.drop(bad_features, axis=1)
data = data.drop(['you_call', 'them_cal'], axis=1)
data.info()


# In[16]:


data = data.drop(['wave'], axis=1)


# In[17]:


data = data.fillna(data.mean())


# In[18]:


data_male = data.query('gender == 1').drop_duplicates(subset=['iid', 'pid'])                                 .drop(['gender'], axis=1)                                 .dropna()
data_female = data.query('gender == 0').drop_duplicates(subset=['iid'])                                   .drop(['gender', 'match', 'int_corr', 'samerace'], axis=1)                                   .dropna()
data_female.columns = data_female.columns + '_f'


# In[19]:


data_female.index = data_female.iid_f
data_female


# In[20]:


data_male.index = data_male.iid
data_male


# In[21]:


data_femaile = data_female.astype({'pid_f':'int'})
data_male = data_male.astype({'pid':'int'})
join_data = data_male.join(data_female, on='pid')
join_data = join_data.dropna()
join_data


# Разбейте датасет на трейн и валидацию. Подберите на валидации оптимальный критерий  информативности. 
# Постройте графики зависимости точности на валидации и трейне от глубины дерева, от минимального числа объектов для сплита. (Т.е должно быть 2 графика, на каждой должны быть 2 кривые - для трейна и валидации)
# Какой максимальной точности удалось достигнуть?

# In[22]:


X = join_data.drop(['match'], axis=1)
y = join_data.match


# In[23]:


np.unique(y, return_counts=True)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[26]:


# Поиск оптимального критерия информативности
criterions = ['gini', 'misclass', 'entropy']
criterion_scores = []
criterion_scores_train = []

for c in criterions:
    dates_clf = MyDecisionTreeClassifier(criterion=c)#, max_depth=best_max_depth, min_samples_split=best_min_samples_split)
    dates_clf.fit(X_train, y_train)
    prediction = dates_clf.predict(X_test)
    acc_test = accuracy_score(prediction, y_test)
    criterion_scores.append(acc_test)
    prediction = dates_clf.predict(X_train)
    acc = accuracy_score(prediction, y_train)
    criterion_scores_train.append(acc)
    print(f"criterion: {c:<8}  train: {round(acc,5)}  test: {round(acc_test,5)}")


# In[27]:


#Самый лучший критерий - misclass 
best_criterion='misclass'


# In[28]:


min_splits = [i for i in range(1, 26)]
min_splits_scores = []
for s in min_splits:
    dates_clf = MyDecisionTreeClassifier(criterion=best_criterion, min_samples_split=s)
    dates_clf.fit(X_train, y_train)
    prediction = dates_clf.predict(X_test)
    acc_test = accuracy_score(prediction, y_test)
    prediction = dates_clf.predict(X_train)
    acc = accuracy_score(prediction, y_train)
    min_splits_scores.append((acc, acc_test))
    print(f"splits: {s:<2}  train: {round(acc,5)}  test: {round(acc_test,5)}")


# In[29]:


import matplotlib.pyplot as plt
plt.plot(min_splits, min_splits_scores)
plt.legend(['train', 'test'])


# In[30]:


# На валидации лучшее значение при сплите от 1 до 4
best_min_samples_split = 4


# In[31]:


max_depths = range(1, 21)
max_depths_scores = []
for d in max_depths:
    dates_clf = MyDecisionTreeClassifier(criterion=best_criterion, max_depth=d)
    dates_clf.fit(X_train, y_train)
    prediction = dates_clf.predict(X_test)
    acc_test = accuracy_score(prediction, y_test)
    prediction = dates_clf.predict(X_train)
    acc = accuracy_score(prediction, y_train)
    max_depths_scores.append((acc, acc_test))
    print(f"depth: {d:<2}  train: {round(acc,5):<7}  test: {round(acc_test,5)}")


# In[32]:


plt.plot(max_depths, max_depths_scores)
plt.legend(['train', 'test'])


# In[33]:


# Берем max_depth равное 4, так как в этой точке имеется небольшой всплеск на валидации
best_max_depth = 8


# In[34]:


best_clf = MyDecisionTreeClassifier(min_samples_split=best_min_samples_split, max_depth=best_max_depth, criterion=best_criterion)
best_clf.fit(X_train, y_train)
best_prediction = best_clf.predict(X_test)
best_acc = accuracy_score(best_prediction, y_test)
# Максимальная точность, которой удалось достичь: 0.8333333333333334
best_acc


# In[35]:


best_clf.tree


# In[36]:


np.unique(best_prediction, return_counts=True)


# Известным фактом является то, что деревья решений сильно переобучаются при увеличении глубины и просто запоминают трейн. 
# Замечаете ли вы такой эффект судя по графикам? Что при этом происходит с качеством на валидации? 
Из графика видно, что начиная с глубины = 9, точность остаётся неизменной. 
На валидации в этом случае достигается максимум, и он тоже статичен.
# ## Находим самые важные признаки (2 балла)
# 
# 

# По построенному дереву  легко понять, какие признаки лучше всего помогли решить задачу. Часто это бывает нужно  не только  для сокращения размерности в данных, но и для лучшего понимания прикладной задачи. Например, Вы хотите понять, какие признаки стоит еще конструировать -- для этого нужно понимать, какие из текущих лучше всего работают в дереве. 

# Самый простой метод -- посчитать число сплитов, где использовался данные признак. Это не лучший вариант, так как по признаку который принимает всего 2 значения, но который почти точно разделяет выборку, число сплитов будет очень 1, но при этом признак сам очень хороший. 
# В этом задании предлагается для каждого признака считать суммарный gain (в лекции обозначено как Q) при использовании этого признака в сплите. Тогда даже у очень хороших признаков с маленьким число сплитов это значение должно быть довольно высоким.  

# Реализовать это довольно просто: создаете словарь номер фичи : суммарный гейн и добавляете в нужную фичу каждый раз, когда используете ее при построении дерева. 

# Добавьте функционал, который определяет значения feature importance. Обучите дерево на датасете Speed Dating Data.
# Выведите 10 главных фичей по важности.

# In[37]:


imps = best_clf.get_feature_importance()
plt.bar(range(imps.shape[0]), imps)


# In[38]:


#imps = np.array(imps)
print("10 main features:", [(i,imps[i]) for i in np.argsort(imps)[::-1][0:10]])


# ## Фидбек (бесценно)

# * Какие аспекты обучения деревьев решений Вам показались непонятными? Какое место стоит дополнительно объяснить?
До конца нпонятно, как строить дерево решений.
Не понимаю, как ещё сильнее ускорить работу дерева и что ещё можно было перенести в numpy.
# ### Ваш ответ здесь

# * Здесь Вы можете оставить отзыв о этой домашней работе или о всем курсе.

# ### ВАШ ОТЗЫВ ЗДЕСЬ
# 
# 
Эта дз вызвала у меня гораздо большие сложности, чем первая и я до конца не уверен в правильности написанного кода.
# In[ ]:




