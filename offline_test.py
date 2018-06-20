from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:\\CTR\\data\\feature\\'

'''
50名之前采用了stacking算法和模型融合，找到了很多强特；
50~100名，挖掘出了强特，但是模型参数还在调试中，还在尝试模型融合。
100名之后，强特还在挖掘中
'''

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
# 0.07889 0.08034 

[1481]	training's binary_logloss: 0.0807668	valid_1's binary_logloss: 0.0788916
'''

def get_data():
    data = pd.read_csv(file_path+'basic_inner_userClick_trick_cvr_cnt_dec.csv')
    data = data.reset_index(drop=True)  # 必须重置索引
    return data

def fea_select(data):
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

def lgb_train_offline():
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
    lgbm = LGBMClassifier(
        task='train',
        boosting_type='gbdt',
        learning_rate=0.01,
        n_estimators=20000,
        max_depth=6,
        subsample=0.4,
        colsample_bytree=0.4,
        reg_lambda=0.8,
        nthread=8,
        seed=128
    )

    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)], eval_metric='binary_logloss', early_stopping_rounds=50)
    y_prob = lgbm.predict_proba(X_val, num_iteration=lgbm.best_iteration_)[:, 1]
    print('result log_loss = {0}'.format(log_loss(y_val, y_prob)))

    fea_score = pd.DataFrame()
    fea_score['feature'] = X_train.columns.tolist()
    fea_score['score'] = list(lgbm.booster_.feature_importance(importance_type='gain'))
    fea_score = fea_score.sort_values(by='score', ascending=False).reset_index(drop=True)
    fea_score.to_csv(file_path+'fea_score.csv', index=None)

    fea_map = pd.Series(data=fea_score['score'].values, index=fea_score['feature'])
    fea_map.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    print('end...')

if __name__ == '__main__':
    lgb_train_offline()
