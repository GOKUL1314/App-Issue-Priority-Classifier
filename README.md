# App Issue Priority Classifier & Care Management Platform

A production-ready Django and Scikit-Learn platform designed for creator e-commerce ecosystems to triage, prioritize, and manage app-related support tickets.

Creators use the platform to build stores and sell products to their contacts via shareable links. When issues occur with the **platform itself** (e.g. broken live links, site builder styles not applying, locked accounts, upload failures, bargain button malfunctions), an intelligent multi-model ML classifier combined with platform safety rules automatically ranks issues by urgency so customer care teams resolve the most critical problems first.

---

## 🌟 Key Features

### 1. Dual Portal Architecture
- **Creator Portal (`/creator/customer-care/`)**:
  - Issue reporting with input validation and character constraints.
  - Platform context integration (`link_status`, `contacts_shared`, `hours_since_link_created`, `device`, `issues_last_30d`) with a toggleable `DEMO_MODE`.
  - Creator privacy: Creators only see **"Complaint received. Ticket #<id>. Our care team will review it."** and status updates. Urgency priorities, confidences, and rule names are strictly kept internal.
- **Care Manager Portal (`/manager/issues/`)**:
  - Prioritized queue ordered strictly by urgency (**High > Medium > Low**) and **oldest waiting first** (ascending creation time).
  - Dynamic waiting timer (e.g. `15m`, `3h 20m`, `2d 4h`).
  - Interactive status & priority update modal with manager correction tracking.
  - One-click CSV export protected against CSV formula injection (`=`, `+`, `-`, `@`, `\t`, `\r`).
  - Interactive Chart.js analytics dashboard featuring the **Manager Agreement Rate** (`final_priority == predicted_priority`).
  - ML training workbench displaying 5-algorithm benchmarks, CV macro-F1, per-class recall, and confusion matrix.

### 2. Strict 3-Tier Urgency Hierarchy
- **High**: Live product link broken for contacts, account locked or hacked, products/data lost or deleted.
- **Medium**: Feature broken but a workaround exists (e.g., photo upload fails, bargain button broken on small lists, live styling glitch).
- **Low**: How-to questions, minor cosmetic issues, feature suggestions, or issues on unpublished draft links.

### 3. Platform Rules (Can Only RAISE)
- **Rule A (Broken Live Link)**: When `app_area == "Product Link"`, `link_status == "Live"`, `contacts_shared >= 1`, and text indicates link not opening / 404 / error, priority is escalated to **High**.
- **Rule B (Account Compromise / Data Loss)**: When text indicates cannot log in, account locked, hacked, someone else logged in, or lost/deleted products or data, priority is escalated to **High**.
- **Strict Invariant**: Rules may only **raise** priority, never lower it (cosmetic downgrade rules are completely eliminated).

### 4. Human-in-the-Loop Feedback Engine
- When a Care Manager overrides an issue's priority in the queue, the issue is tagged with `priority_source="manager"` and timestamped (`corrected_at`).
- During model retraining, manager corrections are automatically extracted and appended as training samples (`extra_df`), continuously adapting the model to human supervisor feedback.

---

## 🧠 Machine Learning Pipeline

- **Features Preprocessor (`ColumnTransformer`)**:
  - **Text (`description`)**: `TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)`
  - **Categorical (`app_area`, `link_status`, `device`)**: `OneHotEncoder(handle_unknown="ignore")`
  - **Numeric (`contacts_shared`, `hours_since_link_created`, `issues_last_30d`)**: `MinMaxScaler()` (enables non-negative support for Naive Bayes)
- **Model Selection**: 5-fold Stratified Cross-Validation on the training split (tie-breaker: higher High-recall).
- **Evaluated Algorithms**:
  1. **Random Forest** (Selected Production Model: **90.47% Macro-F1**, **95.24% High-Recall**)
  2. **Logistic Regression** (90.58% Macro-F1)
  3. **Support Vector Machine (`LinearSVC` with `CalibratedClassifierCV`)** (90.28% Macro-F1)
  4. **Decision Tree** (85.70% Macro-F1)
  5. **Multinomial Naive Bayes** (82.59% Macro-F1)
- **Majority-Class Baseline**: 37.7%

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/GOKUL1314/App-Issue-Priority-Classifier.git
cd App-Issue-Priority-Classifier
pip install -r requirements.txt
```

### 3. Generate Realistic Demo Dataset
Generate the 1,800+ sample creator-platform dataset with context-dependent labels:
```bash
python data/make_demo_data.py
```

### 4. Train 5 ML Algorithms
Train the pipelines, perform 5-fold stratified cross-validation, and persist the best model:
```bash
python ml/train.py
```

### 5. Setup Database & Seed Sample Tickets
Run migrations and seed demo accounts and realistic issues:
```bash
python manage.py makemigrations Creator_Portal
python manage.py migrate
python data/seed_db.py
```

### 6. Run Automated Tests
Execute the full test suite verifying rule invariants, queue sorting, creator privacy, and manager corrections:
```bash
python manage.py test
```

### 7. Start the Development Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser.

---

## 🔑 Default Demo Credentials

| Role | Username | Password | Accessible Endpoints |
| :--- | :--- | :--- | :--- |
| **Creator** | `tech_creator` | `creator123` | `/creator/customer-care/` |
| **Creator (2)** | `game_streamer` | `streamer123` | `/creator/customer-care/` |
| **Care Manager** | `admin` | `admin123` | `/manager/issues/`, `/manager/train/`, `/manager/charts/`, `/manager/export/` |

---

## 📁 Project Structure

```text
app_issue_priority/
├── app_issue_priority/        # Django project settings and root routing
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── Creator_Portal/            # Creator issue submission and authentication
│   ├── models.py              # AppIssue and CreatorAccount models
│   ├── views.py               # Customer care & registration logic
│   └── urls.py
├── Care_Manager_Portal/       # Care management triage, analytics & retraining
│   ├── views.py               # Ranked queue, CSV export, charts & training view
│   ├── tests.py               # Comprehensive platform test suite
│   └── urls.py
├── ml/                        # Pure-Python ML and NLP engine
│   ├── constants.py           # Domain constants, areas, priorities, and features
│   ├── train.py               # 5 ML algorithms training, 5-fold CV & metrics.json
│   ├── predict.py             # score_issue() inference & priority raising rules
│   └── artifacts/             # Serialized best model (.joblib) & metrics summary (.json)
├── data/                      # Dataset generation and seeding utilities
│   ├── make_demo_data.py      # Context-dependent demo dataset generator (>=1500 rows)
│   ├── seed_db.py             # Realistic database seeder
│   └── acceptance_checks.py   # Verification scripts for context and unseen tests
├── Template/htmls/            # Bootstrap 5 responsive UI templates
│   ├── base.html              # Role-aware navigation layout
│   ├── index.html             # Landing portal switcher
│   ├── Creator/               # Creator customer care and auth templates
│   └── Manager/               # Ranked queue, charts, and training workbench
├── requirements.txt           # Python dependencies (Django>=4.2,<6, scikit-learn>=1.2)
└── README.md                  # Project documentation
```

---

## 🛡️ License & Compliance
This software is developed for internal creator triage operations. CSV exports are guarded against formula injection vulnerabilities. Passwords use PBKDF2 with SHA-256 hashing.
