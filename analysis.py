import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy.random import beta as beta_dist

def by_day_explore(df_click_merged):
    #CR by Day
    emails_by_day = pd.DataFrame(df_click_merged['weekday'].value_counts()).reset_index()
    clicks_by_day = pd.DataFrame(df_click_merged.groupby(['weekday']).sum()['clicked']).reset_index()
    emails_by_day.columns = ['day', 'count']
    clicks_by_day.columns = ['day', 'click']
    ctr_by_day = pd.merge(left = emails_by_day, right = clicks_by_day, how = 'inner', on = 'day')
    ctr_by_day['ctr'] = ctr_by_day['click'] / ctr_by_day['count']
    ctr = ctr_by_day[['day', 'count', 'ctr']]
    ctr.head(10)

def by_pp_explore(df_click_merged):
    #CR by Past Purchases
    past_purchase_eda = df_click_merged[['user_past_purchases', 'clicked']]
    past_purchase_eda['counter'] = 1
    grouped_df = past_purchase_eda.groupby('user_past_purchases').sum()
    grouped_df['ctr'] = grouped_df['clicked'] / grouped_df['counter']
    x_axis = range(0, 23)
    plt.plot(x_axis, grouped_df['ctr'])
    plt.xlim(0,15)
    plt.ylim(0, .2)
    plt.xlabel('past_purchases')
    plt.ylabel('ctr')
    plt.show()

def by_hour_explore(df_click_merged):
    #CR by Hour
    hour_eda = df_click_merged[['hour', 'clicked']]
    hour_eda['counter'] = 1
    grouped_df = hour_eda.groupby('hour').sum()
    grouped_df['ctr'] = grouped_df['clicked'] / grouped_df['counter']
    x_axis = range(1, 25)
    plt.plot(x_axis, grouped_df['ctr'])
    plt.xlim(1,24)
    plt.xlabel('hour')
    plt.ylabel('ctr')
    plt.show()

def dummify_categorical(df_click_merged):
    #Dummify Categorical
    df_click_merged['is_short'] = np.where(df_click_merged['email_text'] == 'short_email', 1, 0)
    df_click_merged['is_personalized'] = np.where(df_click_merged['email_version'] == 'personalized', 1, 0)
    df_click_merged['day_of_week_bins'] = df_click_merged['weekday'].apply(convert_days)
    df_click_merged['user_past_purchases'] = pd.qcut(df_click_merged['user_past_purchases'], q = 4, labels = ['past_purchases_q1', 'past_purchases_q2', 'past_purchases_q3', 'past_purchases_q4'])
    df_click_merged['hour'] = pd.cut(df_click_merged['hour'], bins = 4, labels = ['hour_1_6', 'hour_7_12', 'hour_13_18', 'hour_19_24'])

    dfs = df_click_merged, pd.get_dummies(df_click_merged.user_country), pd.get_dummies(df_click_merged.day_of_week_bins), pd.get_dummies(df_click_merged.user_past_purchases), pd.get_dummies(df_click_merged.hour)
    df_categorical = pd.concat(dfs, axis = 1)
    return df_categorical

def split(df_model):
    #Train Test Split for Cross Validation
    v_features = df_model.columns
    v_features = v_features.tolist()
    v_features = v_features[:]
    del v_features[v_features.index('clicked')]

    X = df_model.ix[:, v_features]
    y = df_model.clicked.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=67)
    return X, X_train.values, X_test.values, y_train.values, y_test.values

def baseline_model(X_train, y_train):
    #First Cross-validated models
    lr = LogisticRegression(class_weight='balanced')
    print 'Linear Regresion:\n', KFoldCVScore(lr, X_train, y_train)
    rfc = RandomForestClassifier(n_estimators=250, n_jobs=-1, class_weight='auto')
    print 'Random Forest Classifier:\n', KFoldCVScore(rfc, X_train, y_train)

def KFoldCVScore(model, X_train, y_train):
    #Perform cross validation
    kf = StratifiedKFold(y_train, n_folds=5, shuffle=True)
    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kf:
        model.fit(X_train[train_index], y_train[train_index])
        y_predict = model.predict(X_train[test_index])
        y_true = y_train[test_index]
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print "accuracy:", np.average(accuracies)
    print "precision:", np.average(precisions)
    print "recall:", np.average(recalls)

def SMOTE_sam(X_train, y_train):
    #Increase size of minority class with synthetic samples
    print('Original dataset shape {}'.format(Counter(y_train)))
    sm = SMOTE(ratio = .5)
    X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
    print('Resampled dataset shape {}'.format(Counter(y_train_smote)))
    return X_train_smote, y_train_smote

def convert_days(day_string):
    #Bin day of week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    if day_string in weekdays:
        return 'is_MTWTh'
    elif day_string == 'Friday':
        return 'is_F'
    else:
        return 'is_SaSu'

def plot_importance(clf, X, max_features=20):
    '''Plot feature importance'''
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (X.columns[sorted_idx])[-max_features:]

    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def create_optimum(df_model):
    #subset users based on optimum features
    df_optimum = df_model.copy()
    df_optimum['optimum'] = np.where((df_optimum['is_personalized'] == 1) & (df_optimum['is_short'] == 1) & (df_optimum['is_MTWTh'] == 1) & (df_optimum['hour_7_12'] == 1), 1, 0)
    df_optimum['counter'] = 1
    df_grouped = df_optimum[['clicked', 'optimum', 'counter']].groupby('optimum').sum()
    df_grouped['ctr'] = df_grouped['clicked'] / df_grouped['counter']
    df_grouped.reset_index(inplace=True)
    return df_grouped

def calc_beta_prob(df_grouped, num_samples = 100000):
    #perform bayesian test to simulate future data distributions
    clicks_new = df_grouped[df_grouped['optimum'] == 1]['clicked']
    view_new = df_grouped[df_grouped['optimum'] == 1]['counter']

    clicks_old = df_grouped[df_grouped['optimum'] == 0]['clicked']
    view_old = df_grouped[df_grouped['optimum'] == 0]['counter']

    new_samples = beta_dist(1 + clicks_new, 1 + view_new - clicks_new, num_samples)
    old_samples = beta_dist(1 + clicks_old, 1 + view_old - clicks_old, num_samples)

    return np.mean(new_samples - old_samples > .015)

if __name__ == '__main__':
    df_email = pd.read_csv('email_table.csv')
    df_email_open = pd.read_csv('email_opened_table.csv')
    df_email_click = pd.read_csv('link_clicked_table.csv')

    #Setup DF
    df_email_click['clicked'] = True
    df_click_merged = pd.merge(left = df_email, right = df_email_click, how = 'left')
    df_click_merged.fillna(value = False, inplace = True)

    #EDA Work
    user_choice = raw_input('Show EDA? Y or N:\n')
    if user_choice == 'Y':
        by_hour_explore(df_click_merged)
        by_pp_explore(df_click_merged)
        by_day_explore(df_click_merged)

    #Convert categorical variables into dummy columns based on EDA work
    df_categorical = dummify_categorical(df_click_merged)

    df_model = df_categorical[[u'past_purchases_q1', u'past_purchases_q2', u'past_purchases_q3',
                     u'past_purchases_q4', u'is_short', u'is_personalized', u'ES', u'FR', u'UK',
                     u'US', u'is_F', u'is_MTWTh', u'is_SaSu', u'hour_1_6', u'hour_7_12',
                     u'hour_13_18', u'hour_19_24', u'clicked']]

    #Split data for future cross-validations
    X, X_train, X_test, y_train, y_test = split(df_model)

    print "BASELINE MODEL:"
    baseline_model(X_train, y_train)

    #Deal with class imabalance using SMOTE
    print "\nSMOTE time..."
    X_train_smote, y_train_smote = SMOTE_sam(X_train, y_train)
    print "SMOTE done.\n"

    #RFC chosen as best predictor
    rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight='auto')
    rfc.fit(X_train_smote, y_train_smote)
    y_predict = rfc.predict(X_test)

    print "Final Accuracy:", accuracy_score(y_test, y_predict)
    print "Final Precision:", precision_score(y_test, y_predict)
    print "Final Recall:", recall_score(y_test, y_predict)

    #Plot feature importances to understand contributers to CTR
    print "\nContinuing to Analysis..."
    plot_importance(rfc, X)

    #Isolate emails that met optimum characteristics to maximize CTR
    df_grouped = create_optimum(df_model)
    prob = calc_beta_prob(df_grouped)

    #Use beta dist. to est. probability of seeing similar CTR on future samples
    print "There is a {} percent chance that the optimum model"
    print "would have a greater than 1.5% increase in the CTR.".format(prob)
