from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:\\CTR\\data\\feature\\'

'''
# 0.08119
# 0.08113
# 0.08109
# 0.08106
# 0.08105
# 0.08104
# 0.08048
# 0.08005
# 0.07999 0.08136 
# 0.07979
# 0.07973
# 0.07953 0.08122
# 0.07948
# 0.07922
# 0.07901 0.08083
# 0.07896
# 0.07893 0.08074
'''

def get_data():
    data = pd.read_csv(file_path+'basic_userClick_trick_cvr_cnt_dec.csv')
    data = data.reset_index(drop=True)  # 必须重置索引
    return data

def fea_select(data):
    del data['item_pv_level']
    # del data['context_id_cnt']
    # data= data.drop(data[data.shop_score_service == 0.971369].index.values)
    # data = data.drop(data[data.user_age_level == 1003.4791].index.values)
    return data

def gen_train_val_test(data, offline=True):
    test = data[data.is_trade == -99]
    test_instance = test.instance_id
    test.drop(['is_trade', 'context_date', 'instance_id'], axis=1, inplace=True)
    if offline:
        tr = data[data.is_trade != -99]
        train = tr[tr.context_date <= '2018-09-23 23:59:59']
        val = tr[tr.context_date > '2018-09-23 23:59:59']
        train.drop(['context_date','instance_id'], axis=1, inplace=True)
        val.drop(['context_date','instance_id'], axis=1, inplace=True)
        return train, val, test
    else:
        train = data[data.is_trade != -99]
        train.drop(['context_date','instance_id'], axis=1, inplace=True)
        return train, test, test_instance

def xgb_train_offline():
    print('data process...')
    data = get_data()
    data = fea_select(data)

    id_fea = ['user_id', 'item_id', 'shop_id', 'context_id', 'context_page_id']
    data.drop(id_fea, axis=1, inplace=True) # 暂时不先利用这些id特征

    train, val, test = gen_train_val_test(data, True)

    y_train = train['is_trade']
    X_train = train.drop(['is_trade'], axis=1)

    y_val = val['is_trade']
    X_val = val.drop(['is_trade'], axis=1)

    print('start training...')
    xgb = XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.01,
        n_estimators=10,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=0.005,
        nthread=4,
        seed=128,
        silent=10
    )

    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)], eval_metric='logloss', early_stopping_rounds=50)
    y_prob = xgb.predict_proba(X_val, ntree_limit=xgb.best_ntree_limit)[:, 1]
    print('result log_loss = {0}'.format(log_loss(y_val, y_prob)))

    fea_score = pd.DataFrame()
    fea_score['feature'] = X_train.columns.tolist()
    fea_score['score'] = list(xgb.booster().get_fsocre())
    fea_score = fea_score.sort_values(by='score', ascending=False).reset_index(drop=True)
    fea_score.to_csv(file_path+'fea_score.csv', index=None)

    fea_map = pd.Series(data=fea_score['score'].values, index=fea_score['feature'])
    fea_map.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    print('end...')

if __name__ == '__main__':
    xgb_train_offline()

