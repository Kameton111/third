# Машинное обучние
## Для чего нужна эта математическая модель?
**Задачей этой математической модели является прогноз результата кто купил и тех кто не купил обучние у сайта**

**Прогноз составляется на основе данных взятых с датасета "train.csv"**

**Ссылка на датасет: https://github.com/Kameton111/images/blob/main/train.csv**

## На основе каких данных составляется прогноз?

**Данные пользователей которые были использованы:**

1. Текущее занятие пользователя (школа, университет, работа). 
1. Форма обучения.
1. Главное в людях. (1 — ум и креативность;  2 — доброта и честность; 3 — красота и здоровье;  4 — власть и богатство; 5 — смелость и упорство; 6 — юмор и жизнелюбие).
1. Год начала работы.
1. Год окончания работы.
1. Статус обучения.

## Структура модели?

**Во первых надо установить sklearn.**

***Sklearn - бесплатная библиотека машинного обучения для языка программирования Python.***

**После импортируем все необходимые элементы**

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
```
**После создается две переменные равные нулю**
```
i = 0
aver_result = 0
```
**Затем, очищаем строки от не нужных столбцов:**
```
df = df.drop('id', axis = 1)
df = df.drop('bdate', axis = 1)
df = df.drop('followers_count', axis = 1)
df = df.drop('langs', axis = 1)
df = df.drop('city', axis = 1)
df = df.drop('last_seen', axis = 1)
df = df.drop('occupation_name', axis = 1)
df = df.drop('career_start', axis = 1)
df = df.drop('career_end', axis = 1)
df = df.drop('graduation', axis = 1)
```
**Главная часть математической модели:**
```
def convert(data):
    if type(data)==str():
        return float(data)

while i<5:
    x = df.drop('result', axis = 1)
    y = df['result']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
    aver_result += accuracy_score(y_test, y_pred) * 100
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    i += 1
print('Процент правильно предсказанных исходов:', aver_result/5)
```

## Демонстрация работы модели:

![](https://github.com/Kameton111/images/blob/main/digital_edu.py%20-%20level%20(Workspace)%20-%20Visual%20Studio%20Code%202022-07-25%2020-16-22.gif)

**После активации кода он показывает всю информацию о датафрейме и пять различных прогнозов**

## Использованные библиотеки

**1. Програмная библиотека для анализа данных Pandas**

https://pandas.pydata.org/ **оффициальный сайт Pandas**

**2. Библиотека машинного обучения Sklearn**

https://scikit-learn.org/stable/ **оффициальный сайт Sklearn**
