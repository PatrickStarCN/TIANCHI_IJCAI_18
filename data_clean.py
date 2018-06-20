import pandas as pd

file_path = 'C:\\CTR\\data\\'
train = pd.read_csv(file_path + 'train.csv') # (478138, 27)
# train.drop_duplicates(inplace=True) # 部分重复
test = pd.read_csv(file_path + 'test.csv') # (18371, 26)
print(set(train.columns) - set(test.columns)) # {'is_trade'}
test['is_trade'] = -99
all = pd.concat([train,test], axis=0)

# 缺失值
# import numpy as np
# all.replace(-1, np.nan, inplace=True)
# all.replace('-1', np.nan, inplace=True)
#
# df_null = pd.Series(all.isnull().any())
# print(df_null[df_null == True])

'''
item_brand_id                True 491 将缺失值视为一个类别
item_city_id                 True 283 将缺失值视为一个类别
item_sales_level             True 948 取众数 12

user_gender_id               True 13371 将缺失值视为一个类别
user_age_level               True 982 取众数 1003
user_occupation_id           True 982 将缺失值视为一个类别
user_star_level              True 982 取众数 3006

shop_review_positive_rate    True 7 取众数 1.000000
shop_score_service           True 60 取众数 0.979661
shop_score_delivery          True 60 取众数 0.979589
shop_score_description       True 60 取众数 0.975442

predict_category_property    True 暂不处理，当前特征先不用
'''

# 数值特征进行处理：改为平均值填充
all.item_sales_level.replace(-1, 11.1328, inplace=True)
all.user_age_level.replace(-1,1001.492, inplace=True)
all.user_star_level.replace(-1, 2998.2869, inplace=True)
all.shop_review_positive_rate.replace(-1, 0.994833, inplace=True)
all.shop_score_service.replace(-1, 0.979661, inplace=True)
all.shop_score_delivery.replace(-1, 0.979589, inplace=True)
all.shop_score_description.replace(-1, 0.9748, inplace=True)

all.user_star_level.replace(-1, 1001.4925, inplace=True)
all.user_age_level.replace(-1, 2998.2871, inplace=True)
all.item_sales_level.replace(-1, 11.1329, inplace=True)

train_new = all[all.is_trade != -99]
test_new = all[all.is_trade == -99]
print(train_new.shape)
print(test_new.shape)
train_new.to_csv(file_path + 'train_clean.csv', index=None)
test_new.to_csv(file_path + 'test_clean.csv', index=None)







