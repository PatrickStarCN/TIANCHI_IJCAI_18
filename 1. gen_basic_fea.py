import pandas as pd
import time
import gc

file_path = 'C:\\CTR\\data\\'

def get_data():
    train = pd.read_csv(file_path+'train_clean.csv')  # (478111, 27)
    test = pd.read_csv(file_path+'test_clean.csv')  # (18371, 27)
    data = pd.concat([train, test], axis=0).reset_index(drop=True)  # 必须重置索引
    return data

def process_time_stamp(data):
    lambda_fun = lambda x:time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))
    data['context_date'] = data['context_timestamp'].apply(lambda_fun)
    del data['context_timestamp']
    gc.collect()
    return data

def process_list_fea(data):
    for i in range(3):
        data['item_category_%d' % (i)] = data['item_category_list'].apply( # 从属关系
            lambda x: x.split(";")[i] if len(x.split(";")) > i else '-1'
        )

    sel_property_index=[0,2,5,6,9,12,16,17,19,20,22,23]
    for i in sel_property_index:
        data['item_property_%d' % (i)] = data['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else '-1'
        )

    for i in range(3):
        data['pred_category_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else '-1'
        )

    #del data['item_category_list']
    #del data['item_property_list']
    #del data['predict_category_property']

    del data['item_category_0'] # 商品类目1，常数列，说明是item都是一大类下的
    del data['item_category_2'] # 商品类目3，缺失值太多，不具备区分性

    del data['user_gender_id'] # 可能用户性别和职业，部分是预测值，效果并不好
    del data['user_occupation_id']
    del data['item_pv_level']

    gc.collect()

    return data

if __name__ == '__main__':
    data = get_data()
    data = process_time_stamp(data)
    data = process_list_fea(data)
    data.to_csv(file_path+'basic_data.csv', index=None)