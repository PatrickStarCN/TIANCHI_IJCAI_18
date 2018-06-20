import pandas as pd
import numpy as np

raw_data_path = 'C:\\CTR\\data\\'
feature_data_path ='C:\\CTR\\data\\feature\\'

def process_item_category(x):
    ret = set()
    ret.add(x['item_category_list'].split(';')[1])
    return ret

def process_pred_category(x):
    ret = set()
    for i in range(3):
        if len(str(x['predict_category_property']).split(";")) > i:
            category = str(x['predict_category_property'].split(";")[i]).split(":")[0]
            if category != np.nan:
                ret.add(category)
    return ret

def process_inner_category_num(x):
    if len(set(x['item_category_set']) & set(x['predict_category_set'])) > 0:
        return 1
    else:
        return 0

def process_inner_category_val(x):
    if len(set(x['item_category_set']) & set(x['predict_category_set'])) > 0:
        for val in (set(x['item_category_set']) & set(x['predict_category_set'])):
            return val
    else:
        return '-1'

def gen_inner_category_fea(data):
    '''
    生成交叉类别，数量特征：
    '''
    data['item_category_set'] = data.apply(process_item_category, axis=1) # 必须加axis=1
    data['predict_category_set'] = data.apply(process_pred_category, axis=1)

    data['category_inner_flag'] = data.apply(process_inner_category_num, axis=1)
    data['category_inner_flag'] = data['category_inner_flag'].astype(np.int64)
    print(data['category_inner_flag'].value_counts())

    #data['category_inner_fea'] = data.apply(process_inner_category_val, axis=1)
    #enc = LabelEncoder()
    #data['category_inner_fea'] = enc.fit_transform(data['category_inner_fea'])

    del data['item_category_set']
    del data['predict_category_set']
    return data

if __name__ == '__main__':
    data = pd.read_csv(raw_data_path + 'basic_data.csv')
    data = data.reset_index(drop=True) # 必须重置索引
    data = gen_inner_category_fea(data)
    del data['item_category_list']
    del data['item_property_list']
    del data['predict_category_property']
    data.to_csv(feature_data_path + 'basic_inner.csv', index=None)