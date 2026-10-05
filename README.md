# ML Experiment Lab

> Desktop platform for creating, running, monitoring, analysing and comparing Machine Learning experiments.

## 1. Project Overview

**ML Experiment Lab** is a planned desktop application whose main purpose is to provide a single environment for practical Machine Learning experimentation.

The user should be able to:

1. create an account and sign in;
2. upload and manage datasets;
3. inspect and analyse the data before training;
4. select a Machine Learning task;
5. choose one or more ML models;
6. configure model hyperparameters manually;
7. configure or review preprocessing;
8. start a training run;
9. monitor progress and logs while the model is training;
10. stop a running training job when necessary;
11. evaluate the trained model using relevant metrics;
12. visualise and compare results;
13. save and load trained models;
14. use a saved model for prediction;
15. keep a history of experiments;
16. run benchmarks containing multiple models/runs.

The long-term goal is to make the system a **general-purpose ML experimentation laboratory**, while keeping the diploma-project implementation focused on a realistic and fully working core.

---

## 2. Current Status

### Current milestone

**Milestone 0 — Project Foundation**

Current implemented foundation:

- Python project structure;
- `src`-based package layout;
- PySide6 desktop application skeleton;
- first application window;
- initial dashboard page;
- basic configuration module;
- pytest smoke-test setup;
- Git repository setup;
- GitHub workflow planned for version control and work across multiple computers.

### Current development principle

The project is being built **step by step while the technologies are learned**. Features are not considered complete merely because they are planned; the README distinguishes clearly between current implementation and future scope.

---

## 3. Project Vision

The central idea is:

```text
Dataset
   ↓
Data Analysis
   ↓
ML Task
   ↓
Model Selection
   ↓
Preprocessing
   ↓
Hyperparameters
   ↓
Training
   ↓
Evaluation
   ↓
Visualisation
   ↓
Comparison
   ↓
Save Model
   ↓
Prediction
```

For larger comparisons, the platform should also support:

```text
Benchmark
   ↓
Multiple Models
   ↓
Multiple Training Runs
   ↓
Metrics + Time + Resource Usage
   ↓
Comparison
```

The application is intended to feel like a **laboratory for ML experiments**, not just a collection of unrelated model buttons.

---

## 4. Main Goals

### 4.1 Educational goal

The project should make ML experimentation understandable for beginners while still exposing enough technical information for advanced users.

### 4.2 Practical goal

The platform should reduce the amount of repetitive code required to prepare data, train models, evaluate them and compare results.

### 4.3 Engineering goal

The application should have a modular architecture so that new models, task types and execution strategies can be added without rewriting the entire application.

### 4.4 Diploma-project goal

The final diploma project should demonstrate competence in:

- Python programming;
- Object-Oriented Programming;
- desktop application development;
- REST/API communication;
- databases;
- data processing;
- Machine Learning;
- software architecture;
- testing;
- version control;
- technical documentation.

---

## 5. Target Users

The platform is planned for several user types:

- **Beginners in ML** — need a simple workflow and explanations.
- **Students** — use the platform for learning and experiments.
- **Developers** — compare algorithms, parameters and performance.
- **Teachers** — demonstrate algorithms and evaluation techniques.

The interface should have a simple default mode, with advanced controls available separately.

---

## 6. Product Modes

The project is planned around different levels of control.

### Basic Mode

Designed for users who have minimal ML experience.

The interface should expose the important decisions without overwhelming the user.

### Pro Mode

Planned for more advanced experimentation.

Possible features include:

- detailed hyperparameters;
- custom train/validation/test split;
- cross-validation;
- Grid Search;
- Random Search;
- advanced metrics;
- execution limits;
- additional technical information.

### Developer/Expert Mode

A possible future mode for debugging and technical inspection.

Possible features include:

- raw logs;
- execution configuration;
- API/debug information;
- detailed runtime information.

---

## 7. MVP Scope

The following functionality is considered the **core target for the diploma implementation**.

### User management

- Registration.
- Login.
- User-specific data access.
- Planned authentication providers: email/password, Google and GitHub.

### Dataset management

Initial supported formats:

- CSV;
- XLSX;
- JSON.

The dataset module should provide:

- upload;
- validation;
- preview;
- row/column information;
- data-type inspection;
- missing-value detection;
- basic statistics;
- basic visualisation.

### ML task types

Core tasks:

- Classification;
- Regression;
- Clustering.

Additional task types are planned as extensions.

### Model selection

The initial model library is expected to contain representative algorithms such as:

**Classification**

- Logistic Regression;
- K-Nearest Neighbors;
- Decision Tree;
- Random Forest;
- Support Vector Machine;
- Naive Bayes.

**Regression**

- Linear Regression;
- Polynomial Regression;
- Decision Tree Regressor;
- Random Forest Regressor;
- Support Vector Regression;
- KNN Regressor.

**Clustering**

- K-Means;
- DBSCAN;
- Hierarchical Clustering.

The model system should be designed so that additional algorithms can be added later.

### Hyperparameters

Users should be able to configure model hyperparameters manually.

The UI should eventually provide:

- default values;
- validation;
- allowed ranges where appropriate;
- explanations;
- reset-to-default functionality.

### Training

A training run should:

- run without freezing the desktop UI;
- expose status and progress when possible;
- produce logs;
- allow the user to stop the run;
- end with one of the defined job states such as completed, cancelled or failed.

### Evaluation

The platform should calculate metrics appropriate to the selected task.

Examples:

**Classification**

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- Confusion Matrix.

**Regression**

- MAE;
- MSE;
- RMSE;
- R².

**Clustering**

- Silhouette Score;
- Inertia;
- other appropriate indicators where applicable.

### Visualisation

Planned visualisations include appropriate combinations of:

- bar charts;
- line charts;
- scatter plots;
- histograms;
- correlation matrices;
- confusion matrices;
- model-comparison charts;
- feature importance where supported.

### Experiments

An **Experiment** is the main record that connects:

```text
Dataset
+ Task
+ Preprocessing
+ Model
+ Hyperparameters
+ Training Run
+ Metrics
+ Logs
+ Results
```

The experiment history should make it possible to inspect and compare previous runs.

### Model management

The application should support:

- saving trained models;
- loading saved models;
- using saved models for prediction;
- keeping the relationship between a model and its originating experiment.

### Comparison

Users should be able to compare selected models and experiments using metrics and technical indicators such as training time.

### Benchmark

Benchmarking is planned as a higher-level feature in the core product direction.

A benchmark should be able to run multiple models under comparable conditions and produce a consolidated result.

---

## 8. Long-Term Architecture

The intended architecture is a desktop client connected to an online backend, with a separate ML execution layer.

```text
┌──────────────────────────────┐
│       Desktop Client         │
│            PySide6           │
│                              │
│  UI / Navigation / Charts    │
│  Datasets / Experiments      │
│  Models / Results / Login    │
└──────────────┬───────────────┘
               │ HTTP / API
               ▼
┌──────────────────────────────┐
│        Backend API           │
│           FastAPI            │
│                              │
│ Authentication               │
│ Users / Datasets             │
│ Experiments / Jobs           │
│ Access control               │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌─────────────────┐
│   Database   │  │    ML Engine    │
│              │  │                 │
│ Users        │  │ Preprocessing   │
│ Datasets     │  │ Training        │
│ Experiments  │  │ Evaluation     │
│ Metrics      │  │ Prediction     │
│ Logs         │  │ Model Storage  │
└──────────────┘  └─────────────────┘
```

### Local / Remote execution concept

The planned product direction is hybrid:

**Free/local execution**

```text
Desktop App
   ↓
Local ML Engine
   ↓
User CPU / GPU
```

**Paid/remote execution — future**

```text
Desktop App
   ↓
Internet
   ↓
Backend
   ↓
Remote ML Worker
   ↓
Server CPU / GPU
```

The remote execution model is a future product capability and is not required to make the MVP work.

---

## 9. Technology Stack

### Current / primary choices

- **Python** — main development language.
- **PySide6** — desktop GUI framework.
- **FastAPI** — planned backend API framework.
- **scikit-learn** — primary ML library for the first implementation.
- **Git** — source-code version control.
- **GitHub** — remote repository and collaboration/synchronisation platform.

### Planned technologies

Depending on the final architecture:

- **Pandas** — tabular data processing;
- **NumPy** — numerical operations;
- **Matplotlib** — visualisation;
- **SQLAlchemy** — database access layer;
- **SQLite** — simple development/local database;
- **PostgreSQL** — possible production/server database;
- **pytest** — automated testing.

### Future ML technologies

- PyTorch;
- TensorFlow;
- specialised NLP/CV libraries;
- additional ML frameworks where justified.

---

## 10. Why These Technologies?

### Python

Python is selected because it provides a mature ecosystem for both desktop development and Machine Learning. It also allows the UI, backend and ML implementation to stay within one primary language during the diploma project.

### PySide6

PySide6 is selected as the current desktop UI direction because it provides a mature widget toolkit, layouts, events, signals/slots and model/view patterns that fit a complex desktop application.

### FastAPI

FastAPI is the current backend direction because the application needs an API boundary between the desktop client and online services. It also provides validation and automatic API documentation that are useful during development.

### scikit-learn

scikit-learn is the first ML framework because it contains many standard classical ML algorithms, preprocessing utilities and evaluation tools in a consistent interface. This makes it suitable for building a model library before adding deep-learning frameworks.

### Git/GitHub

Git is used to preserve the history of the project and GitHub acts as the remote copy of the repository. This also allows development to continue from multiple computers.

---

## 11. Data and Processing Pipeline

The intended dataset pipeline is:

```text
Upload
  ↓
Validation
  ↓
Preview
  ↓
Data Analysis
  ↓
Task Definition
  ↓
Preprocessing
  ↓
Train/Test Split
  ↓
Training
  ↓
Prediction
  ↓
Evaluation
  ↓
Results
```

The important design principle is **transparency**: when the application performs preprocessing automatically, it should record and display what was done.

Examples:

```text
Missing values detected
→ median imputation

Categorical feature detected
→ encoding

Numerical features require scaling
→ scaling applied
```

The exact preprocessing strategies will be decided during implementation and testing.

---

## 12. Experiment Concept

The central data object of the application is the **Experiment**.

A simplified experiment may contain:

```text
Experiment
├── Dataset
├── Task
├── Target / input definition
├── Preprocessing configuration
├── Model
├── Hyperparameters
├── Training Run
├── Metrics
├── Visualisations
├── Logs
└── Saved Model
```

This structure is intended to support reproducibility and comparison.

---

## 13. Training Job Concept

Training should be treated as a separate job rather than as an action that blocks the entire interface.

Possible states:

```text
QUEUED
   ↓
RUNNING
   ↓
COMPLETED
```

Alternative endings:

```text
RUNNING → CANCELLED
RUNNING → FAILED
```

A future implementation may also support pausing/resuming for algorithms where the underlying framework allows it safely.

---

## 14. Logging

Logging is an explicit project requirement.

Possible events include:

```text
Dataset uploaded
Experiment created
Training started
Preprocessing started
Model training started
Evaluation started
Training completed
Training cancelled
Training failed
Model saved
Prediction completed
```

Logs should help both the user and the developer understand what happened during an experiment.

---

## 15. Security Direction

Security is part of the design, especially because the application is expected to use an online backend.

Planned principles:

- passwords are never stored as plain text;
- authenticated API access;
- authorization based on the logged-in user;
- users can access only permitted datasets and experiments;
- uploaded files are validated;
- sensitive configuration is stored outside committed source code;
- arbitrary user Python code is **not** part of the MVP.

Running arbitrary uploaded code would require sandboxing and stronger isolation and is therefore treated as a separate future architecture problem.

---

## 16. Future Extensions

These features are intentionally separated from the MVP so they do not endanger the diploma deadline.

### High-value extensions

- AutoML;
- Grid Search;
- Random Search;
- Cross Validation;
- improved benchmark orchestration;
- remote/server training;
- automatically generated reports;
- plugin-based model registration.

### ML extensions

- neural networks;
- PyTorch;
- TensorFlow;
- NLP;
- computer vision;
- time-series modelling;
- anomaly detection.

### Product extensions

- Google/GitHub authentication;
- experiment sharing;
- public/private datasets;
- community experiments;
- cloning/versioning experiments;
- AI assistant;
- natural-language experiment commands;
- richer resource monitoring;
- automatic updates;
- cross-platform installers.

---

## 17. Project Scope Rule

The project follows this rule:

> **A complete and stable core is more important than a large number of unfinished features.**

The architecture should support future growth, but the diploma implementation must first provide a coherent end-to-end workflow.

Priority order:

```text
1. Application foundation
2. Dataset workflow
3. ML workflow
4. Training + evaluation
5. Results + comparison
6. Model persistence + prediction
7. Testing + stability
8. Optional extensions
```

---

## 18. Development Method

The project is developed together with the learning process.

For each major technology or concept, the intended workflow is:

```text
Learn the concept
      ↓
Small example
      ↓
Understand the mechanism
      ↓
Apply it to ML Experiment Lab
      ↓
Test the feature
      ↓
Commit the change
```

The objective is not only to make the application work, but to understand and be able to explain every major part of it.

---

## 19. Git Workflow

GitHub is the remote source of truth for the project.

### First setup on a new computer

```powershell
py -3.13 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m ml_experiment_lab
```

### Daily workflow

```powershell
git pull
# work on the project
git status
git add .
git commit -m "description of the change"
git push
```

The virtual environment is local to each computer and should not be committed to Git.

Sensitive files such as `.env` and API credentials must also remain outside the repository.

---

## 20. Repository Structure

Current structure:

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
│       └── ui/
│           ├── __init__.py
│           ├── main_window.py
│           │
│           └── pages/
│               ├── __init__.py
│               └── dashboard_page.py
│
├── tests/
│   └── test_smoke.py
│
├── pyproject.toml
├── README.md
├── MILESTONES.md
└── .gitignore
```

The structure will expand as the application grows. ML, database, API and storage modules will be added when their corresponding development milestones begin.

---

## 21. Development Roadmap

### Phase 0 — Foundation

- project setup;
- Python environment;
- Git/GitHub;
- PySide6 application shell;
- first window;
- tests.

### Phase 1 — Desktop UI

- main layout;
- sidebar/navigation;
- dashboard;
- pages and reusable UI components;
- Basic/Pro UI concept.

### Phase 2 — Backend

- FastAPI service;
- API structure;
- health endpoint;
- configuration;
- authentication foundation.

### Phase 3 — Database

- users;
- datasets;
- experiments;
- training runs;
- metrics;
- logs;
- saved models metadata.

### Phase 4 — Dataset Module

- upload;
- validation;
- parsing;
- preview;
- statistics;
- visualisations.

### Phase 5 — ML Engine

- task abstraction;
- model abstraction;
- preprocessing;
- classification;
- regression;
- clustering;
- evaluation.

### Phase 6 — Training Manager

- background jobs;
- status tracking;
- progress;
- cancellation;
- logging;
- resource monitoring where practical.

### Phase 7 — Experiments and Models

- experiment persistence;
- model save/load;
- prediction;
- experiment history.

### Phase 8 — Comparison and Benchmarking

- model comparison;
- charts;
- benchmark execution;
- result summaries.

### Phase 9 — Quality

- testing;
- error handling;
- security review;
- performance review;
- UI polishing;
- packaging.

### Phase 10 — Optional Extensions

Only after the core is stable:

- AutoML;
- hyperparameter search;
- cross-validation;
- remote training;
- deep learning;
- reports;
- AI assistant;
- other extensions.

---

## 22. Planned Four-Month Strategy

The expected development period is approximately **4 months**, with about **5–10 hours per week** available.

A simplified priority strategy is:

```text
Month 1
Foundation + UI + Backend + Database

Month 2
Datasets + ML Engine + First Models

Month 3
Training + Metrics + Experiments + Models

Month 4
Comparison + Benchmark + Testing + Packaging + Documentation
```

Optional features are added only when the core path is stable.

---

## 23. Quality Criteria for the MVP

The MVP is considered successful when a user can complete an end-to-end workflow similar to:

```text
Login
  ↓
Upload Dataset
  ↓
Inspect Dataset
  ↓
Choose Task
  ↓
Choose Model(s)
  ↓
Configure Parameters
  ↓
Start Training
  ↓
Monitor Logs / Progress
  ↓
Stop if necessary
  ↓
Evaluate
  ↓
View Charts + Metrics
  ↓
Compare
  ↓
Save Experiment
  ↓
Save Model
  ↓
Load Model
  ↓
Predict
```

The application must remain understandable and usable throughout this workflow.

---

## 24. What Is Already Done vs Planned

### Already done

- initial project structure;
- Python package;
- PySide6 application skeleton;
- initial dashboard;
- basic test structure;
- Git/GitHub setup workflow.

### In planning/design

- complete UI;
- authentication;
- backend API;
- database;
- dataset management;
- preprocessing pipeline;
- ML engine;
- training manager;
- evaluation;
- experiment persistence;
- model persistence;
- comparison;
- benchmarking.

### Future

- remote/cloud training;
- AutoML and search strategies;
- deep learning;
- NLP/CV/time-series;
- AI assistant;
- sharing/community features;
- advanced packaging and cross-platform support.

---

## 25. Important Design Principles

### Separation of concerns

UI, backend, ML logic, data access and storage should remain separated.

### Modularity

Adding a new ML algorithm should require a small, well-defined change rather than a rewrite of unrelated modules.

### Reproducibility

Important experiment settings should be recorded so that results can be understood and repeated.

### Transparency

Automatic operations such as preprocessing should be visible to the user.

### Stability first

Core features must be tested and reliable before optional features are added.

### Learn while building

Every technical decision should be understandable to the developer and explainable during the diploma defence.

---

## 26. Next Development Step

**Current next step: Milestone 1 — Build the real desktop application shell.**

The next implementation work should introduce:

- the main application layout;
- sidebar navigation;
- dashboard structure;
- reusable page navigation;
- the first real UI components;
- the architecture required for later Datasets, Experiments, Models and Benchmarks pages.

At this stage, the ML engine is intentionally not implemented yet. We first establish a clean desktop foundation and learn the PySide6 concepts required for it.

---

## 27. Documentation

The project should maintain separate documentation for:

- project plan and analysis;
- technical architecture;
- API documentation;
- database design;
- ML methodology;
- testing;
- user manual;
- final diploma documentation.

These documents should evolve together with the implementation.

---

## 28. Project Status Legend

- `[DONE]` — implemented and tested.
- `[IN PROGRESS]` — currently being developed.
- `[PLANNED]` — agreed project scope, not implemented yet.
- `[OPTIONAL]` — useful extension after the core is stable.
- `[FUTURE]` — long-term product direction.

---

## 29. Current Status Summary

**Project:** ML Experiment Lab  
**Stage:** Foundation  
**Primary language:** Python  
**Desktop:** PySide6  
**Backend direction:** FastAPI  
**ML direction:** scikit-learn first  
**Repository:** Git + GitHub  
**Development target:** working desktop MVP followed by optional extensions  
**Main philosophy:** learn the technology while building the real system, step by step.
