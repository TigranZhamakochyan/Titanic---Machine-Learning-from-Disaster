# Titanic Survival Prediction

My solution to Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.

**Best Kaggle score: 0.80143** (335 of 418 passengers correct)

My best model isn't a machine learning model. After trying many tuned models, a set of three simple rules based on who travelled together scored highest.

## The final model

Passengers are placed in travel groups if they match on any of these links:

1. They share a ticket.
2. They have the same surname, class, port, and family size.
3. They have the same surname, class, and port, and their ticket numbers differ only in the last digit.

Three rules then make each prediction:

| Passenger | Prediction |
|---|---|
| Adult man | Dies |
| Woman | Survives, unless all the known women and boys in her group died |
| Boy | Dies, unless all the known women and boys in his group survived |

Boys are identified by the title "Master". Adult men are left out of the group outcomes because most died regardless of what happened to their families.

## My process

### 1. Machine learning models

I started with passenger details such as sex, class, age, and fare, then compared logistic regression, random forest, XGBoost, Naive Bayes, and other models. I added features such as titles and family size, and tuned the models.

My best result from this stage was V13, a random forest combined with two family-based rules, which scored 0.78468.

### 2. A problem with validation

V13 scored 83.6% in cross-validation but only 78.47% on Kaggle. V15, a logistic regression using travel groups, reached 84.6% in cross-validation, but only 78.95% on Kaggle.

Better cross-validation scores weren't translating into much better Kaggle scores, so I stopped adding complexity.

### 3. Starting again from a simple rule

I rebuilt the project from scratch, starting with the simplest group-based rule. It scored 83.5% in cross-validation, lower than V15, but **0.79904** on Kaggle, higher than any of my machine learning models.

### 4. Testing small improvements

I tested each new idea on its own and decided in advance what result would mean keeping it.

- **Fare rule (rejected):** predicting death for certain 3rd-class women raised cross-validation to 84.3%, but lowered the Kaggle score to 0.79665.
- **Nearby tickets (kept):** linking relatives with nearly matching ticket numbers changed one prediction and raised the Kaggle score to **0.80143**.

## Results

| Model | Cross-validation | Kaggle score |
|---|---|---|
| Gender rule | 78.7% | Not submitted |
| V13: random forest + family rules | 83.6% | 0.78468 |
| V15: logistic regression + travel groups | 84.6% | 0.78947 |
| Group rule | 83.5% | 0.79904 |
| Group rule + fare rule | 84.3% | 0.79665 |
| **Group rule + nearby tickets** | **83.6%** | **0.80143** |

## What I learned

- **Simple models can win.** With only 891 training passengers, complex models can easily fit noise.
- **Understanding the problem matters.** Knowing that families tended to share an outcome helped more than tuning did.
- **Cross-validation has limits.** It was useful for catching leakage, but it couldn't reliably separate models this close.
- **Data leakage is easy to introduce.** My test hides the answers for the passengers being scored, and a safety check confirms that nothing leaks.
- **Decide before you submit.** Testing one fixed idea at a time helps avoid fitting the model to the leaderboard.

## Limitations

The final score beats 0.80 by only one passenger, so the last improvement could be down to chance. The model also can't identify adult men who survived, and it's weaker for passengers with no known group members.

## Repository contents

| File | Description |
|---|---|
| `01_machine_learning_models.ipynb` | My machine learning experiments, from the first models through V15 |
| `02_rule_based_model.ipynb` | The final rule-based model |

## How to run it

1. Install the packages for the final model:

   ```bash
   pip install kagglehub pandas numpy scikit-learn
   ```

2. Sign in to Kaggle and accept the competition rules.
3. Run `02_rule_based_model.ipynb` from top to bottom. It creates `submission_near_ticket.csv`.

The competition data isn't included in this repository. The notebooks download it through `kagglehub`.

The machine learning notebook needs extra packages, including `xgboost`, `catboost`, `matplotlib`, and `seaborn`.
