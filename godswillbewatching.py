# coding: utf-8

import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier


# Step one, load the data.
train_ds = pd.read_csv("./train.csv")
test_ds = pd.read_csv("./test.csv")


# Get gender: male is 1 female is 0
def get_sex(sex):
    if sex == "male":
        return 1
    else:
        return 0


# Where they came from? (Southamptom, Cherbourg, Queenstown). Whe guess Southamptom for the unkown.
def get_embarked(embarked):
    if embarked in ['C']:
        return 1
    elif embarked in ['Q']:
        return 2
    else:
        return 0


# Fix missing age values with median
def fix_age(age):
    if math.isnan(age):
        return 28
    else:
        return age


# Fix 0 or NaN fares
def fix_price(fare):
    if fare == 0 or math.isnan(fare):
        return 14.45
    else:
        return fare


# Family inside:
def family_inside(family):
    if family > 0:
        return 1
    else:
        return 0


# How much they paid for the fare?
def get_status(fare):
    if fare >= 31:
        return 0
    elif 8 < fare < 31:
        return 1
    else:
        return 2


# Create a mapping of the title_ids
def get_title(name):
    title = name.split(' ')[1]
    if title == 'Mr.':
        return 0
    elif title == 'Mrs.':
        return 1
    elif title == 'Miss.':
        return 2
    elif title == 'Master.':
        return 3
    elif title == 'Don.':
        return 4
    else:
        return 5


# Fit the dataset
def fit_ds(ds):
    ds['sex'] = ds['Sex'].apply(get_sex)
    ds['title'] = ds['Name'].apply(get_title)
    ds['price'] = ds['Fare'].apply(fix_price)
    ds['age'] = ds['Age'].apply(fix_age)
    ds['status'] = ds['price'].apply(get_status)
    ds['family'] = ds['SibSp'].apply(family_inside)
    ds['from'] = ds['Embarked'].apply(get_embarked)
    return ds


# We generate fitted datasets
train_fds = fit_ds(train_ds)
test_fds = fit_ds(test_ds)

X_train = train_fds\
    .drop("Survived", axis=1)\
    .drop("Name", axis=1)\
    .drop("Ticket", axis=1)\
    .drop("Cabin", axis=1)\
    .drop("Sex", axis=1)\
    .drop("Embarked", axis=1)\
    .drop("Age", axis=1)\
    .drop("Fare", axis=1)

Y_train = train_fds["Survived"]

random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, Y_train)

test_fds = test_fds.drop("Name", axis=1)\
    .drop("Ticket", axis=1)\
    .drop("Cabin", axis=1)\
    .drop("Sex", axis=1)\
    .drop("Embarked", axis=1)\
    .drop("Age", axis=1)\
    .drop("Fare", axis=1)

prediction = random_forest.predict(test_fds)



submission = pd.DataFrame({
    "Dude": test_ds["Name"],
    "Survived": prediction
})

submission.to_csv("survivedornot.csv", index=False)
