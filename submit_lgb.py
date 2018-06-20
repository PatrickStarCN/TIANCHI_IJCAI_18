from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'E:\\CVR\\data\\feature\\'

def get_data():
    data = pd.read_csv(file_path+'basic_inner_userClick_trick_cvr_cnt_dec.csv')
    data = data.reset_index(drop=True)  # 必须重置索引
    return data

def gen_train_val_test(data, offline=True):
    test = data[data.is_trade == -99]
    test_instance = test.instance_id
    test.drop(['is_trade', 'context_date', 'instance_id'], axis=1, inplace=True)
    if offline == True:
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

def fea_select(data):
    return data

def lgb_make_submission():
    print('data process...')
    data = get_data()
    data = fea_select(data)

    # 暂时不先利用这些id特征
    id_fea = ['user_id', 'item_id', 'shop_id', 'context_id', 'context_page_id']
    data.drop(id_fea, axis=1, inplace=True)

    train, test, test_instance = gen_train_val_test(data, False)

    y_train = train['is_trade']
    X_train = train.drop(['is_trade'], axis=1)

    print('start training...')
    lgbm = LGBMClassifier(
        task='train',
        boosting_type='gbdt',
        learning_rate=0.01,
        n_estimators=1481,
        max_depth=6,
        subsample=0.4,
        colsample_bytree=0.4, # 0.0804237（线上：0.08034）
        reg_lambda=0.8,
        nthread=8,
        seed=128
    )

    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='binary_logloss')
    y_prob = lgbm.predict_proba(X_train)[:, 1]
    print('result log_loss = {0}'.format(log_loss(y_train, y_prob)))

    y_submit = lgbm.predict_proba(test)[:, 1]
    sub = pd.DataFrame()
    sub['instance_id'] = test_instance
    sub['predicted_score'] = y_submit
    sub.to_csv('E:\\CVR\\data\\result\\20180417.txt', sep=" ", index=False)

    fea_score = pd.DataFrame()
    fea_score['feature'] = X_train.columns.tolist()
    fea_score['score'] = list(lgbm.booster_.feature_importance(importance_type='gain'))
    fea_score = fea_score.sort_values(by='score', ascending=False).reset_index(drop=True)
    fea_score.to_csv(file_path + 'fea_score.csv', index=None)

    fea_map = pd.Series(data=fea_score['score'].values, index=fea_score['feature'])
    fea_map.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    print('end...')

if __name__ == '__main__':
    lgb_make_submission()

