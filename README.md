# ML Experiment Lab

> Desktop платформа за провеждане, наблюдение, анализиране и сравняване на Machine Learning експерименти.

---

## Съдържание

1. [Какво е проектът](#1-какво-е-проектът)
2. [Защо съществува платформата](#2-защо-съществува-платформата)
3. [Как работи системата като цяло](#3-как-работи-системата-като-цяло)
4. [Какво е готово и какво е планирано](#4-какво-е-готово-и-какво-е-планирано)
5. [Обхват на MVP](#5-обхват-на-mvp)
6. [Бъдещи разширения](#6-бъдещи-разширения)
7. [Архитектура](#7-архитектура)
8. [Основни части на приложението](#8-основни-части-на-приложението)
9. [Потребителски поток](#9-потребителски-поток)
10. [Datasets](#10-datasets)
11. [Подготовка на данните](#11-подготовка-на-данните)
12. [Machine Learning задачи](#12-machine-learning-задачи)
13. [Model Library](#13-model-library)
14. [Hyperparameters](#14-hyperparameters)
15. [Training](#15-training)
16. [Оценяване и метрики](#16-оценяване-и-метрики)
17. [Визуализация](#17-визуализация)
18. [Experiments](#18-experiments)
19. [Benchmarks](#19-benchmarks)
20. [Запазване и използване на модели](#20-запазване-и-използване-на-модели)
21. [Локално и отдалечено обучение](#21-локално-и-отдалечено-обучение)
22. [Потребители, вход и достъп](#22-потребители-вход-и-достъп)
23. [Backend и API](#23-backend-и-api)
24. [База данни и съхранение](#24-база-данни-и-съхранение)
25. [Логове и обработка на грешки](#25-логове-и-обработка-на-грешки)
26. [Сигурност](#26-сигурност)
27. [Структура на кода](#27-структура-на-кода)
28. [Какво е Git и GitHub в този проект](#28-какво-е-git-и-github-в-този-проект)
29. [Разработка: инсталиране и стартиране](#29-разработка-инсталиране-и-стартиране)
30. [Работа на втори компютър](#30-работа-на-втори-компютър)
31. [Как друг разработчик да клонира и подобрява проекта](#31-как-друг-разработчик-да-клонира-и-подобрява-проекта)
32. [Как крайният потребител ще инсталира приложението](#32-как-крайният-потребител-ще-инсталира-приложението)
33. [Стартиране на приложението](#33-стартиране-на-приложението)
34. [Тестване](#34-тестване)
35. [Принципи за разработка](#35-принципи-за-разработка)
36. [План за разработка](#36-план-за-разработка)
37. [Критерии за завършен MVP](#37-критерии-за-завършен-mvp)
38. [Известни ограничения](#38-известни-ограничения)
39. [Структура на дипломния проект](#39-структура-на-дипломния-проект)
40. [Често срещани проблеми](#40-често-срещани-проблеми)
41. [Бъдещо развитие](#41-бъдещо-развитие)
42. [Речник на термините](#42-речник-на-термините)
43. [Статус и следваща стъпка](#43-статус-и-следваща-стъпка)

---

# 1. Какво е проектът

**ML Experiment Lab** е планирано и постепенно разработвано desktop приложение за експериментиране с Machine Learning.

Основната идея е човек да може да работи с ML чрез един графичен интерфейс, вместо за всяка задача да пише отделна Python програма от нулата.

Платформата трябва да позволи:

- създаване на потребителски профил;
- качване и управление на собствени datasets;
- анализиране на данните преди обучение;
- избор на тип Machine Learning задача;
- избор на един или повече алгоритми;
- ръчно настройване на hyperparameters;
- избор или преглед на preprocessing стъпките;
- стартиране на обучение;
- наблюдение на training процеса;
- преглед на live logs;
- спиране на обучение при необходимост;
- изчисляване на подходящи метрики;
- визуализиране на резултатите;
- сравняване на модели;
- запазване на експерименти;
- запазване и зареждане на модели;
- използване на вече обучен модел за prediction;
- изпълнение на benchmark върху няколко модела.

Дългосрочната идея е системата да стане **ML лаборатория**, в която експериментът е основната единица работа.

---

# 2. Защо съществува платформата

При обучение на Machine Learning модел често трябва да се изпълнят много повтарящи се задачи:

1. зареждане на dataset;
2. проверка на данните;
3. почистване;
4. preprocessing;
5. разделяне на данните;
6. избор на алгоритъм;
7. настройване на параметри;
8. обучение;
9. оценяване;
10. визуализация;
11. сравнение с други модели;
12. запазване на резултатите.

Без специализирана платформа тези действия често се реализират чрез множество отделни Python скриптове или Notebook клетки.

ML Experiment Lab обединява тези действия в един workflow.

### Основна практическа полза

Потребителят трябва да може да зададе:

> „Имам този dataset и искам да проверя кои от избраните модели работят най-добре и при какви настройки.“

и системата да превърне това в подреден и проследим експеримент.

### Основна образователна полза

Платформата трябва да бъде разбираема и за начинаещи. В идеалния случай тя не само показва число като `Accuracy = 0.94`, а позволява на потребителя да разбере:

- какво е направено с данните;
- кой модел е използван;
- какви са параметрите му;
- как е оценен;
- колко време е отнело обучението;
- какво означава полученият резултат.

---

# 3. Как работи системата като цяло

Основният ML workflow е:

```text
Dataset
   ↓
Анализ на данните
   ↓
Избор на ML задача
   ↓
Избор на модел/модели
   ↓
Preprocessing
   ↓
Hyperparameters
   ↓
Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Visualisation
   ↓
Comparison
   ↓
Save Experiment / Model
   ↓
Използване на модела за Prediction
```

За benchmark:

```text
Benchmark
   ↓
Избран Dataset
   ↓
Избрани Models
   ↓
Training Runs
   ↓
Metrics
   ↓
Training Time
   ↓
Resource Usage
   ↓
Comparison
   ↓
Benchmark Result
```

---

# 4. Какво е готово и какво е планирано

Този раздел е важен, защото **README не трябва да представя планирана функционалност като вече реализирана**.

## 4.1. Реализирано към текущия момент

Към текущия етап проектът има основа за разработка:

- Python package структура;
- `src`-based организация;
- PySide6 desktop application skeleton;
- основен application entry point;
- главен прозорец;
- първоначална Dashboard страница;
- базов configuration модул;
- pytest smoke test setup;
- `.gitignore`;
- `pyproject.toml`;
- Git repository;
- GitHub remote workflow.

Това означава, че **основата е създадена**, но основните ML функционалности все още се изграждат поетапно.

## 4.2. Планирано за MVP

MVP означава **Minimum Viable Product** — най-малката версия, която е достатъчно пълна, за да представлява завършена платформа.

MVP включва:

- потребители;
- login/register;
- datasets;
- CSV/XLSX/JSON;
- data analysis;
- classification;
- regression;
- clustering;
- model selection;
- hyperparameters;
- preprocessing;
- training;
- progress;
- logs;
- stop;
- evaluation;
- metrics;
- visualisations;
- experiments;
- model persistence;
- prediction;
- comparison;
- benchmark;
- база данни;
- backend API;
- основна сигурност.

## 4.3. Бъдещо

Функции като AutoML, Deep Learning, NLP, Computer Vision, remote GPU training и AI assistant са описани като разширения и не са задължителни за първата напълно работеща версия.

---

# 5. Обхват на MVP

## 5.1. Потребители

Системата трябва да има индивидуални акаунти.

Минималният функционален набор е:

- регистрация;
- вход;
- изход;
- достъп до собствени данни;
- защита на ресурсите между потребители.

Планирани методи за вход:

- email + password;
- Google;
- GitHub.

## 5.2. Dataset management

Минимално:

- upload;
- проверка на типа файл;
- preview;
- брой редове;
- брой колони;
- типове на колоните;
- missing values;
- основна статистика;
- basic visualisation.

## 5.3. ML задачи

Първоначално:

- Classification;
- Regression;
- Clustering.

## 5.4. Training

Потребителят трябва да може да:

- избере модел;
- зададе parameters;
- стартира training;
- наблюдава прогрес;
- вижда logs;
- прекрати training.

## 5.5. Evaluation

След training системата трябва да показва резултатите чрез подходящи metrics и visualisations.

## 5.6. Experiment history

Всеки experiment трябва да запазва достатъчно информация, за да може по-късно да бъде прегледан и сравнен.

---

# 6. Бъдещи разширения

Разширенията се добавят **след стабилно работещо MVP**.

## 6.1. AutoML

Автоматично тестване на множество модели и конфигурации.

## 6.2. Grid Search

Автоматично проверяване на предварително зададени комбинации от hyperparameters.

## 6.3. Random Search

Автоматично изпробване на случайни комбинации от параметри в определено пространство.

## 6.4. Cross Validation

По-надеждна оценка чрез K-Fold и подобни техники.

## 6.5. Neural Networks

Добавяне на модели чрез PyTorch или TensorFlow.

## 6.6. Computer Vision

Работа с image datasets и модели за изображения.

## 6.7. NLP

Работа с текстови datasets.

## 6.8. Time Series

Поддръжка на времеви редове.

## 6.9. Remote Training

Изпращане на тежък training job към отдалечен сървър.

## 6.10. AI Assistant

Помощ при разбиране на резултатите, настройването на модели и конфигурирането на експерименти.

## 6.11. Natural Language Control

Команди от типа:

> „Сравни Random Forest и SVM върху последния ми dataset.“

## 6.12. Reports

Автоматично генериране на отчет.

## 6.13. Sharing

Споделяне и cloning на experiments.

## 6.14. Community

Публични datasets, experiments и benchmarking резултати.

## 6.15. Plugin system

Добавяне на нови модели като независими модули.

---

# 7. Архитектура

Основната архитектурна идея е отделяне на интерфейса от бизнес логиката и ML изчисленията.

```text
┌───────────────────────────────┐
│         Desktop Client        │
│            PySide6            │
└───────────────┬───────────────┘
                │
          HTTP / JSON
                │
                ▼
┌───────────────────────────────┐
│          Backend API          │
│            FastAPI            │
└───────────────┬───────────────┘
                │
        ┌───────┴─────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│   Database    │  │  Job Manager  │
└───────────────┘  └───────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │    ML Engine   │
                  │  scikit-learn  │
                  └────────────────┘
```

## 7.1. Desktop Client

Отговаря за:

- интерфейса;
- навигацията;
- input формите;
- таблиците;
- графиките;
- показването на training status;
- показването на logs;
- действията на потребителя.

Desktop клиентът не трябва да съдържа целия ML engine вътре в UI кода.

## 7.2. Backend

Отговаря за:

- API;
- authentication;
- authorization;
- управление на jobs;
- достъп до database;
- metadata;
- remote execution при бъдещ режим.

## 7.3. ML Engine

Отговаря за:

- loading на data;
- preprocessing;
- model creation;
- training;
- prediction;
- evaluation;
- serialization;
- model metadata.

---

# 8. Основни части на приложението

## 8.1. Dashboard

Главният екран трябва да показва информация за:

- последни experiments;
- последни datasets;
- бърз старт;
- текущи training jobs;
- базова статистика.

## 8.2. Datasets

Управление на datasets.

## 8.3. Experiments

История на извършените ML експерименти.

## 8.4. Models

Запазени модели и операции върху тях.

## 8.5. Benchmarks

Групови сравнения на модели.

## 8.6. Settings

Настройки на приложението, профила и бъдещи execution preferences.

---

# 9. Потребителски поток

Примерен Basic flow:

```text
Отваряне на приложението
        ↓
Login / Register
        ↓
Dashboard
        ↓
New Experiment
        ↓
Upload Dataset
        ↓
Dataset Analysis
        ↓
Избор на Task
        ↓
Избор на Model
        ↓
Parameters
        ↓
Preprocessing
        ↓
Training
        ↓
Results
        ↓
Comparison
        ↓
Save Model
```

Потребителят трябва да може да се върне назад, да промени настройка и да създаде нов experiment, без да губи стария.

---

# 10. Datasets

## 10.1. Поддържани формати в първата версия

- CSV;
- XLSX;
- JSON.

## 10.2. Какво трябва да се проверява

При upload системата трябва да проверява поне:

- разширение;
- MIME/type когато е наличен;
- дали файлът може да бъде прочетен;
- структурата му;
- дали съдържа данни;
- дали има проблеми с колоните.

## 10.3. Dataset preview

Потребителят трябва да може да види първите редове, колоните и базови характеристики.

## 10.4. Dataset analysis

Примерна информация:

```text
Rows: 150
Columns: 5
Numeric columns: 4
Categorical columns: 1
Missing values: 3
```

Допълнително:

- mean;
- median;
- standard deviation;
- min/max;
- unique values;
- distribution information.

---

# 11. Подготовка на данните

Machine Learning моделът обикновено не може просто да получи произволен необработен dataset и да работи правилно.

Затова имаме preprocessing слой.

## 11.1. Missing values

Възможни стратегии:

- премахване на редове;
- премахване на колони;
- запълване с mean;
- запълване с median;
- запълване с най-честа стойност;
- специални стратегии при определени задачи.

## 11.2. Categorical data

Категориалните стойности трябва да бъдат преобразувани в числова форма, когато избраният модел изисква това.

Например:

```text
red
blue
red
```

може да бъде кодирано чрез подходящ encoder.

## 11.3. Scaling

Някои модели са чувствителни към мащаба на features.

Възможни стратегии:

- StandardScaler;
- MinMaxScaler;
- други подходящи трансформации.

## 11.4. Train/Test split

Минималната система трябва да има отделни training и testing данни.

По-късно могат да се добавят:

- validation set;
- K-Fold Cross Validation.

## 11.5. Data leakage

Системата трябва да бъде проектирана така, че preprocessing параметрите да се научават от training данните, а не от тестовата информация.

Това е важен принцип за честна оценка.

---

# 12. Machine Learning задачи

## 12.1. Classification

При classification целта е да се определи клас.

Пример:

```text
Входни данни → Model → Class A / B / C
```

Примери за алгоритми:

- Logistic Regression;
- KNN;
- Decision Tree;
- Random Forest;
- SVM;
- Naive Bayes.

## 12.2. Regression

При regression целта е да се предскаже числова стойност.

Примери:

- Linear Regression;
- Polynomial Regression;
- Decision Tree Regressor;
- Random Forest Regressor;
- SVR;
- KNN Regressor.

## 12.3. Clustering

При clustering няма предварително зададен клас; алгоритъмът търси групи в данните.

Примери:

- K-Means;
- DBSCAN;
- Hierarchical Clustering.

---

# 13. Model Library

Model Library е централният модул за наличните алгоритми.

## 13.1. Основен принцип

Всички модели трябва да имат общо поведение, доколкото техният тип го позволява.

Концептуално:

```text
Base Model Contract
├── create
├── train
├── predict
├── evaluate
├── save
└── load
```

## 13.2. Защо е важен общият интерфейс

Без общ договор всеки алгоритъм би изисквал различна логика в UI и backend.

С общ интерфейс:

```text
UI
 ↓
Model Service
 ↓
Selected Model
```

Така добавянето на нов алгоритъм е по-лесно.

## 13.3. Първоначален моделeн набор

### Classification

- Logistic Regression;
- KNN;
- Decision Tree;
- Random Forest;
- SVM;
- Naive Bayes.

### Regression

- Linear Regression;
- Polynomial Regression;
- Decision Tree Regressor;
- Random Forest Regressor;
- SVR;
- KNN Regressor.

### Clustering

- K-Means;
- DBSCAN;
- Hierarchical Clustering.

---

# 14. Hyperparameters

Hyperparameter е настройка на алгоритъма, която се задава преди или около training процеса, а не се научава директно от training data по начина, по който моделът научава своите вътрешни параметри.

Примери:

```text
Random Forest
n_estimators = 100
max_depth = 10
```

```text
KNN
n_neighbors = 5
```

Потребителят трябва да може да задава тези стойности ръчно.

## 14.1. UI изисквания

Всеки параметър трябва да има:

- име;
- input control;
- default value;
- type validation;
- допустими граници когато е приложимо;
- кратко обяснение.

## 14.2. Basic срещу Pro

Basic режимът може да показва само най-съществените параметри.

Pro режимът показва повече контрол.

---

# 15. Training

Training е процесът, при който избраният алгоритъм използва training данните, за да изгради или адаптира модел.

## 15.1. Основни състояния на training job

```text
QUEUED
  ↓
RUNNING
  ↓
COMPLETED
```

Алтернативни състояния:

```text
RUNNING → CANCELLED
RUNNING → FAILED
```

## 15.2. Защо training не трябва да блокира UI

Някои модели могат да обучават достатъчно дълго. Ако training се изпълнява директно в основния UI thread, приложението може да спре да реагира.

Затова training трябва да бъде отделен execution task/process според конкретната архитектура.

## 15.3. Live progress

Когато е възможно, интерфейсът трябва да показва:

- status;
- elapsed time;
- logs;
- progress;
- ресурсно използване.

Не всеки алгоритъм има истински процент прогрес. В такъв случай UI не трябва да измисля фалшив прогрес, а трябва да показва честен статус.

## 15.4. Stop

Потребителят трябва да може да поиска прекратяване на job.

Прекратяването трябва да бъде реализирано по безопасен начин, така че приложението да не се повреди и database state да остане консистентен.

---

# 16. Оценяване и метрики

Една от най-важните функции е да не сравняваме всички модели с една и съща metric независимо от задачата.

## 16.1. Classification

Основни метрики:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- Confusion Matrix.

### Accuracy

Дял на правилно класифицираните примери.

### Precision

Каква част от предсказаните положителни случаи действително са положителни.

### Recall

Каква част от реално положителните случаи са открити.

### F1-score

Комбинира Precision и Recall чрез хармонично средно.

### Confusion Matrix

Показва как реалните класове се разпределят спрямо предсказаните.

## 16.2. Regression

Основни метрики:

- MAE;
- MSE;
- RMSE;
- R².

### MAE

Средна абсолютна грешка.

### MSE

Средна квадратична грешка.

### RMSE

Корен от MSE.

### R²

Показател за това доколко моделът обяснява вариацията в target стойността.

## 16.3. Clustering

Основни показатели:

- Silhouette Score;
- Inertia при алгоритми, при които е приложимо.

## 16.4. Технически показатели

Допълнително може да се записват:

- training time;
- prediction time;
- CPU usage;
- RAM usage;
- GPU usage при бъдеща GPU поддръжка;
- размер на модела.

---

# 17. Визуализация

Визуализацията е част от анализа, а не само декоративен елемент.

## 17.1. Dataset visualisations

Възможни:

- histogram;
- scatter plot;
- box plot;
- correlation matrix;
- distribution plots.

## 17.2. Model visualisations

В зависимост от задачата:

- Confusion Matrix;
- ROC Curve;
- Precision-Recall Curve;
- Feature Importance;
- Learning Curve;
- Residual plots;
- cluster visualisation.

## 17.3. Comparison charts

Например:

```text
Model              F1       Training Time
Random Forest      0.95     1.20 s
SVM                0.93     0.40 s
Logistic Regression0.91     0.12 s
```

Графиката трябва да позволява бързо сравнение, но числовите стойности също трябва да остават налични.

---

# 18. Experiments

Experiment е основната единица на платформата.

Един experiment описва конкретно изпълнение на ML задача.

## 18.1. Какво съдържа един experiment

Минимално:

```text
Experiment
├── User
├── Dataset
├── Task
├── Target
├── Preprocessing configuration
├── Model
├── Hyperparameters
├── Training Run
├── Metrics
├── Resource measurements
├── Logs
└── Result state
```

## 18.2. Защо experiment е отделен обект

Това позволява:

- история;
- повторение;
- сравнение;
- анализ;
- reproducibility;
- бъдещо cloning/versioning.

## 18.3. Experiment history

Потребителят трябва да може да види стари експерименти и да отвори подробностите им.

---

# 19. Benchmarks

Benchmark е група от сравними изпълнения.

Пример:

```text
Benchmark: Iris

Models:
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM
```

Резултат:

```text
                 Accuracy   F1      Time
Logistic         0.94       0.94    0.10 s
KNN              0.95       0.95    0.08 s
Decision Tree    0.93       0.93    0.03 s
Random Forest    0.96       0.96    0.80 s
SVM              0.97       0.97    0.15 s
```

Benchmark трябва да използва сравними условия:

- един и същ dataset;
- ясно описан split;
- известен random seed, когато е приложимо;
- еднакви правила за evaluation.

---

# 20. Запазване и използване на модели

След training потребителят трябва да може да запази модела.

Conceptually:

```text
Training
   ↓
Trained Model
   ↓
Save
   ↓
Model Storage
```

По-късно:

```text
Load Model
   ↓
Prediction
```

## 20.1. Prediction

Потребителят може да даде нови данни и да поиска prediction.

В зависимост от модела системата може да показва:

- prediction;
- вероятности;
- confidence-related information, когато моделът предоставя такава и тя е смислена.

Важно е UI да не представя probability като универсална и автоматично надеждна „сигурност“ на модела.

---

# 21. Локално и отдалечено обучение

Това е част от дългосрочната архитектурна идея.

## 21.1. Free / Local mode

При локалния режим training се извършва на компютъра на потребителя.

```text
Desktop App
    ↓
Local ML Engine
    ↓
CPU / RAM / GPU на потребителя
```

Предимства:

- не изисква платен cloud training;
- добър за малки и средни datasets;
- лесен за демонстрация;
- данните могат да останат локални.

Недостатъци:

- зависи от хардуера;
- много тежките задачи могат да бъдат бавни.

## 21.2. Paid / Remote mode

Бъдеща идея:

```text
Desktop App
    ↓
Internet
    ↓
Backend
    ↓
Remote ML Worker
    ↓
CPU / GPU на сървъра
```

Това ще позволи по-тежки изчисления.

Търговската логика, плащанията и реалната cloud инфраструктура не са основна част от дипломния MVP.

---

# 22. Потребители, вход и достъп

## 22.1. Authentication

Authentication отговаря на въпроса:

> „Кой е този потребител?“

Планирани методи:

- email/password;
- Google;
- GitHub.

## 22.2. Authorization

Authorization отговаря на въпроса:

> „Има ли този потребител право да извърши това действие?“

Пример:

Потребител A не трябва да може произволно да отвори private dataset на потребител B.

## 22.3. Multi-user

Платформата е проектирана за повече от един потребител.

Всяка собственост на ресурс трябва да се свързва с потребителски идентификатор.

---

# 23. Backend и API

Основният backend framework е планиран като **FastAPI**.

## 23.1. Какво е API

API е интерфейс, чрез който различни части на системата комуникират.

Пример:

```text
Desktop
   ↓
POST /api/experiments
   ↓
Backend
```

Backend може да върне JSON:

```json
{
  "id": 42,
  "status": "queued"
}
```

## 23.2. Планирани API области

```text
/api/auth
/api/users
/api/datasets
/api/experiments
/api/models
/api/jobs
/api/results
/api/benchmarks
/api/logs
```

Точният API contract ще се фиксира при техническото проектиране.

---

# 24. База данни и съхранение

Планираните основни обекти са:

```text
User
Dataset
Experiment
TrainingRun
Metric
Model
Log
Benchmark
```

Conceptual relationship:

```text
User
 │
 ├── Dataset
 │
 ├── Experiment
 │      │
 │      ├── TrainingRun
 │      ├── Metric
 │      └── Log
 │
 └── Model
```

## 24.1. Database

По време на разработка може да се започне с SQLite за по-малка инфраструктура.

При реален multi-user server deployment е възможен PostgreSQL.

Точният избор се фиксира след архитектурно тестване.

## 24.2. File storage

Database не е задължително да съдържа целия бинарен dataset или model file.

По-подходяща е схемата:

```text
Database
→ metadata

File Storage
→ dataset/model files
```

---

# 25. Логове и обработка на грешки

## 25.1. Application logs

Например:

```text
Application started
User logged in
Dataset selected
```

## 25.2. Training logs

Пример:

```text
[12:31:02] Loading dataset...
[12:31:03] Preprocessing...
[12:31:04] Training Random Forest...
[12:31:08] Evaluation...
[12:31:08] Training complete.
```

## 25.3. Error logs

Грешките трябва да се записват с достатъчно информация за debugging.

## 25.4. User-facing errors

Потребителят не трябва да получава само технически stack trace.

Вместо това UI трябва да показва разбираемо съобщение, например:

> „Файлът не може да бъде прочетен като CSV. Проверете формата и разделителя на колоните.“

Подробният технически error може да остане в log системата.

---

# 26. Сигурност

Минималните изисквания включват:

- password hashing;
- authentication;
- authorization;
- input validation;
- file validation;
- ограничаване на достъпа до user-owned resources;
- защита на sensitive configuration;
- HTTPS при production server.

## 26.1. Sensitive данни

Не трябва да се commit-ват:

```text
passwords
API keys
secret tokens
production credentials
```

Вместо това се използват environment variables или secret-management механизми.

## 26.2. Произволен Python код

Потребителският произволен Python код **не е част от MVP**.

Причината е, че изпълнението на произволен код на server изисква sandboxing и допълнителни security ограничения.

---

# 27. Структура на кода

Основната идея е:

```text
ML_Experiment_Lab/
│
├── src/
│   └── ml_experiment_lab/
│       ├── __init__.py
│       ├── __main__.py
│       ├── main.py
│       ├── config.py
│       │
│       ├── ui/
│       ├── core/
│       ├── data/
│       ├── ml/
│       ├── api/
│       ├── database/
│       └── services/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── api/
│   └── ml/
│
├── docs/
├── pyproject.toml
├── README.md
├── MILESTONES.md
└── .gitignore
```

Не всички папки трябва да бъдат създадени веднага. Те ще се появяват, когато бъде реализирана съответната част.

## 27.1. Принцип на разделение

Не трябва да имаме един файл от хиляди редове, в който се намират:

- UI;
- database;
- ML training;
- authentication;
- API.

Всеки модул има ясна отговорност.

---

# 28. Какво е Git и GitHub в този проект

## Git

Git е система за version control.

Той пази история на промените по проекта.

Пример:

```text
Commit 1
↓
Commit 2
↓
Commit 3
↓
Commit 4
```

## GitHub

GitHub е онлайн място, в което може да съхраняваме Git repository-то.

```text
Компютър
   ↓ git push
GitHub
   ↓ git pull
Друг компютър
```

## Важни Git понятия

### Working directory

Файловете, по които в момента работиш.

### Staging area

Промените, подготвени за commit.

### Commit

Записана версия на избраните промени.

### Remote

Отдалечено repository, например GitHub.

### `origin`

Името по подразбиране за основния remote.

### `main`

Основният branch на проекта.

---

# 29. Разработка: инсталиране и стартиране

Този раздел описва **разработческата среда**, а не готовия installer за краен потребител.

## 29.1. Необходими инструменти

За текущата разработка са необходими:

- Windows като основна разработваща среда;
- Python;
- Git;
- IDE/editor, например VS Code;
- интернет при използване на GitHub и online backend.

Python и Git са разработвачески инструменти. Те не са задължително нужни на крайния потребител на бъдещия инсталиран `.exe` вариант.

## 29.2. Клониране на проекта

Първият път на нов компютър:

```powershell
git clone https://github.com/VeselinSv/ML-Experiment-Lab.git
```

След това:

```powershell
cd ML-Experiment-Lab
```

## 29.3. Създаване на virtual environment

```powershell
py -m venv .venv
```

### Какво е `.venv`

Това е отделна Python среда за конкретния проект.

Тя позволява различните проекти да имат различни версии на библиотеките и да не си пречат.

## 29.4. Активиране

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

След activation често ще видиш:

```text
(.venv) PS C:\...\ML-Experiment-Lab>
```

## 29.5. Инсталиране на зависимостите

```powershell
python -m pip install --upgrade pip
pip install -e .
```

`pyproject.toml` описва зависимостите на проекта.

## 29.6. Стартиране

```powershell
python -m ml_experiment_lab
```

Ако foundation milestone е непокътнат, трябва да се стартира desktop прозорецът.

## 29.7. Стартиране на тестовете

```powershell
pytest
```

Точният test command може да бъде разширен по-късно.

---

# 30. Работа на втори компютър

Това е една от основните причини да използваме GitHub.

## 30.1. Първи път

На втория компютър:

```powershell
git clone https://github.com/VeselinSv/ML-Experiment-Lab.git
cd ML-Experiment-Lab
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## 30.2. Следващите пъти

Преди работа:

```powershell
git pull
```

Работиш.

След това:

```powershell
git status
git add .
git commit -m "описание на промяната"
git push
```

## 30.3. Важна забележка за `.venv`

`.venv` не се качва в GitHub.

Причината е, че virtual environment е локална среда за конкретния компютър.

На всеки нов компютър тя се създава отново.

Това не означава, че трябва ръчно да инсталираш всяка библиотека. `pyproject.toml` описва зависимостите.

---

# 31. Как друг разработчик да клонира и подобрява проекта

Това е отделно от потребителя, който просто иска да използва готовото приложение.

## 31.1. Разработчикът първо инсталира

- Git;
- Python;
- IDE.

## 31.2. Клонира repository

```powershell
git clone https://github.com/VeselinSv/ML-Experiment-Lab.git
```

## 31.3. Създава средата

```powershell
cd ML-Experiment-Lab
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## 31.4. Стартира и тества

```powershell
python -m ml_experiment_lab
pytest
```

## 31.5. Прави промени

Пример:

```text
feature: Dataset page
```

## 31.6. Commit

```powershell
git add .
git commit -m "feat: add dataset page"
```

## 31.7. Push

При права за repository-то:

```powershell
git push
```

При по-сериозна работа бъдещият workflow може да използва отделен branch и pull request.

---

# 32. Как крайният потребител ще инсталира приложението

Тук има важна разлика между **текущия developer build** и **бъдещия production build**.

## 32.1. Текущ етап

Проектът все още се разработва като Python приложение.

Следователно в текущия foundation е нормално developer да има:

- Python;
- `.venv`;
- dependencies;
- source code.

## 32.2. Краен вариант

Целта е да има installer, например:

```text
ML-Experiment-Lab-Setup.exe
```

Крайният потребител трябва да може:

1. да стартира installer;
2. да избере папка;
3. да инсталира приложението;
4. да отвори ML Experiment Lab.

Потребителят **не трябва да инсталира Python, Git или pip ръчно**.

## 32.3. Packaging

По време на разработката ще изберем и документираме конкретен packaging инструмент, например PyInstaller или друг подходящ инструмент.

Това ще се реши след като основното приложение е стабилно.

## 32.4. Зависимости при крайния потребител

Не трябва да се изисква:

```text
pip install PySide6
pip install pandas
pip install scikit-learn
```

Тези dependency-и се пакетират с приложението или се управляват от runtime/deployment системата.

---

# 33. Стартиране на приложението

## 33.1. Developer mode

```powershell
python -m ml_experiment_lab
```

## 33.2. Final installed mode

След реализиране на installer идеята е потребителят да използва:

```text
Start Menu
   ↓
ML Experiment Lab
```

или desktop shortcut.

Няма да му е необходим PowerShell за нормална работа.

---

# 34. Тестване

Тестовете са част от разработката, а не последна задача.

## 34.1. Unit tests

Тестват малки самостоятелни части.

Пример:

```text
data validation
```

## 34.2. Integration tests

Проверяват дали няколко модула работят правилно заедно.

## 34.3. API tests

Проверяват backend endpoint-и.

## 34.4. ML tests

Проверяват:

- входни данни;
- shapes;
- preprocessing;
- training;
- prediction;
- metric calculation.

## 34.5. UI tests

Проверяват критичните user flows, когато техническият избор на framework и tooling го позволява.

## 34.6. Manual acceptance tests

Преди release трябва да се мине през сценарий като:

```text
Register
→ Login
→ Upload dataset
→ Configure experiment
→ Train
→ Stop / Complete
→ Results
→ Save
→ Reload
→ Prediction
```

---

# 35. Принципи за разработка

## 35.1. Не започваме с всичко

Първо завършваме основния workflow.

## 35.2. Проектът се изгражда едновременно с учене.

## 35.3. Малки commits

Всеки commit трябва да има смислено съдържание.

Примери:

```text
chore: initialize project
feat: add main window navigation
feat: add dataset validation
feat: add classification engine
fix: prevent UI blocking during training
```

## 35.4. Отделни отговорности

UI не трябва да съдържа database SQL логика.

ML engine не трябва да зависи директно от конкретен button.

## 35.5. Разширяемост

Нов ML model трябва да може да бъде добавен без пренаписване на цялото приложение.

## 35.6. Честна оценка

Не се използват неподходящи metrics само защото изглеждат добре.

---

# 36. План за разработка

Проектът е планиран за приблизително 4 месеца.

## Фаза 1 — Foundation

- Git;
- Python environment;
- project structure;
- desktop skeleton;
- tests.

## Фаза 2 — UI архитектура

- MainWindow;
- sidebar;
- page navigation;
- Dashboard;
- общ UI стил.

## Фаза 3 — Backend основа

- FastAPI;
- API structure;
- configuration;
- authentication foundation.

## Фаза 4 — Database

- users;
- datasets;
- experiments;
- models;
- jobs;
- logs.

## Фаза 5 — Dataset module

- upload;
- validation;
- reading;
- preview;
- statistics.

## Фаза 6 — ML engine

- model abstraction;
- classification;
- regression;
- clustering.

## Фаза 7 — Preprocessing

- missing values;
- encoding;
- scaling;
- train/test split.

## Фаза 8 — Training manager

- job states;
- background execution;
- progress;
- cancellation;
- logs.

## Фаза 9 — Evaluation

- metrics;
- confusion matrix;
- regression metrics;
- clustering metrics.

## Фаза 10 — Visualization

- charts;
- comparison;
- model result pages.

## Фаза 11 — Persistence

- experiment storage;
- model storage;
- load;
- prediction.

## Фаза 12 — Benchmark

- multiple models;
- multiple runs;
- summary;
- comparison.

## Фаза 13 — Testing

- unit;
- integration;
- ML;
- acceptance.

## Фаза 14 — Packaging

- build;
- installer;
- clean-machine test.

## Фаза 15 — Documentation

- technical documentation;
- screenshots;
- UML;
- database diagram;
- testing results;
- final user guide.

---

# 37. Критерии за завършен MVP

MVP се приема за завършен, когато:

- приложението стартира;
- основният desktop интерфейс работи;
- потребителят може да се регистрира и/или да влезе според реално реализирания auth режим;
- dataset може да бъде качен;
- dataset може да бъде анализиран;
- избира се ML задача;
- избира се модел;
- задават се параметри;
- training започва;
- UI не блокира по време на дълъг training;
- logs са видими;
- training може да бъде прекратен;
- резултатите се изчисляват;
- metrics се визуализират;
- модели могат да се сравняват;
- experiment се записва;
- модел може да се запази и зареди;
- prediction е възможен;
- benchmark има работещ основен flow;
- основните грешки са обработени;
- проектът може да бъде стартиран от чиста developer среда по документираните инструкции;
- основните test сценарии минават.

---

# 38. Известни ограничения

Следните ограничения са умишлени:

1. Не се реализират всички съществуващи Machine Learning алгоритми в първата версия.
2. Deep Learning не е задължителен за MVP.
3. Cloud GPU training е future scope.
4. Production billing не е част от diploma MVP.
5. Произволно изпълнение на user Python code не е част от MVP.
6. Public community функционалността не е задължителна.
7. Cross-platform build ще бъде разглеждан след стабилизиране на основната Windows версия.

---

# 39. Структура на дипломния проект

Техническата разработка може да бъде основа за следната структура на дипломната работа:

## Част I — Теоретична

- Artificial Intelligence;
- Machine Learning;
- supervised learning;
- unsupervised learning;
- classification;
- regression;
- clustering;
- preprocessing;
- model evaluation;
- overfitting/underfitting;
- hyperparameters;
- benchmarking.

## Част II — Анализ

- проблем;
- цел;
- потребители;
- functional requirements;
- non-functional requirements;
- use cases;
- ограничения;
- risk analysis.

## Част III — Проектиране

- architecture;
- modules;
- database;
- API;
- UI;
- ML pipeline;
- experiment model.

## Част IV — Реализация

- technologies;
- code structure;
- desktop client;
- backend;
- database;
- ML engine;
- training manager;
- evaluation;
- visualisation.

## Част V — Тестване

- test cases;
- unit tests;
- integration tests;
- ML tests;
- acceptance tests;
- performance observations.

## Част VI — Експерименти и резултати

- datasets;
- benchmark experiments;
- comparison;
- conclusions.

## Част VII — Заключение и бъдещо развитие

- постигнати цели;
- ограничения;
- future work.

---

# 40. Често срещани проблеми

## Проблем: `python` не се разпознава

Проверка:

```powershell
py --version
```

и:

```powershell
python --version
```

Ако едната команда работи, използваме нея според средата.

## Проблем: PowerShell не позволява activation

При такава грешка трябва да се провери PowerShell execution policy и конкретните инструкции за developer средата. Не трябва механично да се изключва security политиката без разбиране на последствията.

## Проблем: `git push` е rejected с `fetch first`

Това означава, че remote repository съдържа commit, който локалното repository няма.

Обичайният безопасен flow е:

```powershell
git status
git pull --rebase origin main
git push origin main
```

Ако има conflict, първо се преглежда `git status`, след което conflict-ът се разрешава ръчно.

**Не трябва да се използва `git push --force` без конкретна причина и разбиране на последствията.**

---

# 41. Бъдещо развитие

Когато MVP е стабилно, проектът може да се развие в следните направления:

```text
MVP
 ↓
Pro Mode
 ↓
AutoML
 ↓
Advanced Search
 ↓
Deep Learning
 ↓
Remote Training
 ↓
AI Assistant
 ↓
Sharing / Community
```

Архитектурата трябва да позволява тези функции да се добавят постепенно.

---

# 42. Речник на термините

**Algorithm** — метод/процедура, чрез която се решава ML задача.

**Model** — обучен или параметризиран модел, който може да прави prediction.

**Dataset** — набор от данни.

**Feature** — входна характеристика на пример.

**Target** — стойността, която моделът се опитва да предскаже при supervised learning.

**Training** — процес на обучение на модела.

**Prediction** — резултатът от използването на модел върху входни данни.

**Metric** — количествен показател за качеството на модела.

**Hyperparameter** — настройка на алгоритъма, която се задава при конфигуриране на training процеса.

**Preprocessing** — подготовка и преобразуване на данните.

**Experiment** — конкретно описан ML run с dataset, configuration, training и резултати.

**Benchmark** — група сравними experiments/runs.

**Backend** — сървърната част на приложението.

**Frontend/Desktop Client** — интерфейсът, с който потребителят работи.

**API** — интерфейс за комуникация между софтуерни компоненти.

**Authentication** — установяване на самоличност.

**Authorization** — проверка на права.

**Git** — система за version control.

**GitHub** — онлайн услуга за Git repositories.

**Commit** — записана версия на промени.

**Remote** — отдалечено repository.

**Virtual environment** — изолирана Python среда за проект.

---

# 43. Статус и следваща стъпка

## Текущ статус

**Milestone 0 — Foundation**

Основата е създадена и repository-то е публикувано в GitHub:

```text
https://github.com/VeselinSv/ML-Experiment-Lab
```

## Следваща реална техническа стъпка

След основата ще преминем към **Milestone 1 — Desktop UI architecture**.

Целта е да се изгради:

```text
MainWindow
   ↓
Sidebar
   ↓
Page Navigation
   ├── Dashboard
   ├── Datasets
   ├── Experiments
   ├── Models
   ├── Benchmarks
   └── Settings
```

Преди всяка значима стъпка се изучават необходимите concepts и след това се прилагат директно в проекта.

---

# Приложение A — Базов developer workflow

```text
1. Отвори проекта
        ↓
2. Активирай .venv
        ↓
3. git pull
        ↓
4. Работи
        ↓
5. pytest
        ↓
6. git status
        ↓
7. git add .
        ↓
8. git commit -m "..."
        ↓
9. git push
```

---

# Приложение B — Базов workflow за нов компютър

```powershell
git clone https://github.com/VeselinSv/ML-Experiment-Lab.git
cd ML-Experiment-Lab
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m ml_experiment_lab
```

---

# Приложение C — Минимален Git cheat sheet

```powershell
# Провери статуса
git status

# Вземи новите промени
git pull

# Добави промени
git add .

# Направи commit
git commit -m "описание"

# Качи към GitHub
git push

# Виж историята
git log --oneline

# Виж свързаните remote-и
git remote -v
```

---

# Приложение D — Какво не се качва в Git

Примерно:

```text
.venv/
__pycache__/
*.pyc
.env
секрети
локални datasets при нужда от privacy
локални build файлове
```

Точното съдържание се определя от `.gitignore`.

---

# Приложение E — Основна философия на ML Experiment Lab

Платформата не е просто:

```text
„Избери модел → натисни Train“
```

Тя е:

```text
Разбери данните
      ↓
Избери задача
      ↓
Избери модел
      ↓
Разбери параметрите
      ↓
Обучи
      ↓
Наблюдавай
      ↓
Оцени
      ↓
Сравни
      ↓
Запази
      ↓
Повтори / подобри експеримента
```

Това е основната концепция, около която трябва да се развива целият проект.
