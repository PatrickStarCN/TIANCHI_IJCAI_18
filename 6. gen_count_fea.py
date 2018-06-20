from tqdm import tqdm
import pandas as pd

file_path = 'C:\\CTR\\data\\feature\\'

def gen_ID_global_count(data, feature):
    feature_count_sum = pd.DataFrame(
        data.groupby(feature).size()
    ).reset_index().rename(columns={0:feature+'_cnt'})
    return feature_count_sum

def gen_global_count(data, stats_features):
    for feature in tqdm(stats_features):
        feature_count_sum = gen_ID_global_count(data, feature)
        data = data.merge(feature_count_sum,'left',[feature])
    return data

def gen_category1_fea_nunique(data):
    fea_all = [ ('item_category_1', 'item_id'),
                ('item_category_1', 'shop_id')]
    for fea1,fea2 in fea_all:
        temp=data.groupby(fea1)[fea2].nunique().reset_index()
        temp.columns=[fea1,fea1+'_'+fea2+'_nunique']
        data=data.merge(temp,on=fea1,how='left')
    return data

if __name__ =='__main__':

    data = pd.read_csv(file_path + 'basic_inner_userClick_trick_cvr.csv')
    data = data.reset_index(drop=True)  # 必须重置索引

    stats_features = ['item_id', 'shop_id', 'context_id', 'context_page_id', 'item_city_id']
    data = gen_global_count(data, stats_features)
    data = gen_category1_fea_nunique(data)

    data.to_csv(file_path+'basic_inner_userClick_trick_cvr_cnt.csv', index=None)


