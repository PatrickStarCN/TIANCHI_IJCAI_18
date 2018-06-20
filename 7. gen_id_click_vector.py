import pandas as pd
import numpy as np

file_path = 'C:\\CTR\\data\\feature\\'

def gen_CountVector_ID_user_clicks(data, user_info, desc_fea, dummy_fea, drop_na=False):
    desc_data = data[['user_id', desc_fea]]

    prefix_name = desc_fea + '_desc_by_' + dummy_fea
    sub_user_info = pd.get_dummies(user_info[['user_id', dummy_fea]], columns=[dummy_fea], prefix=prefix_name)
    if drop_na:
        sub_user_info.drop([prefix_name + '_-1'], axis=1, inplace=True)
    desc_data = pd.merge(desc_data, sub_user_info, 'left', 'user_id')

    dummy_features = sub_user_info.columns.tolist()
    dummy_features.remove('user_id')

    # 合并后主表：按照shop_id分组，然后dummy特征求和（比如年龄：实际上就是去描述，当前shop_id值，不同年龄点击数向量）
    ID_describe_feature = desc_data[[desc_fea] + dummy_features].groupby([desc_fea], as_index=False).sum()
    return ID_describe_feature

def add_CountVector_ID_user_clicks(data, desc_fea, dummy_fea):
    user_info = data[['user_id', 'user_age_level', 'user_star_level']]
    user_info['user_age_level'] = user_info['user_age_level'] % 1000
    user_info['user_age_level'] = user_info['user_age_level'].astype(np.int64)
    user_info['user_star_level'] = user_info['user_star_level'] % 1000
    user_info['user_star_level'] = user_info['user_star_level'].astype(np.int64)

    desc_vec = gen_CountVector_ID_user_clicks(data, user_info, desc_fea, dummy_fea)
    data = pd.merge(data, desc_vec, how='left', on=desc_fea)
    return data
    
if __name__ == '__main__':
    data = pd.read_csv(file_path + 'basic_inner_userClick_trick_cvr_cnt.csv')
    data = data.reset_index(drop=True)  # 必须重置索引
    data = add_CountVector_ID_user_clicks(data, 'shop_id', 'user_age_level') # 18,25
    data.to_csv(file_path + 'basic_inner_userClick_trick_cvr_cnt_dec.csv', index=None)
    
