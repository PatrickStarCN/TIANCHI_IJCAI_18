'''
不同的上下文页面id具有不同的转化计算方式:
如第一次点击算转化; 最后一次点击算转化; 总点击算转化，
分析并构造相应描述特征，提升模型预测精度。

这里的统计特征，都是按每天，而不是多天累计，避免了考虑新用户，冷启动问题。
'''
from tqdm import tqdm
import pandas as pd
import numpy as np

file_path = 'C:\\CTR\\data\\feature\\'

def apply_cv_trick(row):
    if row['click_count'] <= 1:
        return 0
    elif row['click_first'] > 0:
        return 1
    elif row['click_last'] > 0:
        return 2
    else:
        return 3

def gen_day_cv_trick(df,sel_fea):
    """
    输入指定点击天数df，然后统计，在当前点击日期（天）：
    1. 每个用户id，点击上下文页面id，总次数，设标志位
    2. 每个用户id，点击上下文页面id，第一次，设标志位
    3. 每个用户id，点击上下文页面id，最后一次，设标志位
    然后，根据指定apply规则，计算转化
    """
    click_count = df.groupby(['user_id', sel_fea]).size().reset_index() # 全部点击次数标志位  
    click_count.rename(columns={0: 'click_count'}, inplace=True) 
    click_count = click_count[['user_id', sel_fea, 'click_count']] # 只取这三列
    df = pd.merge(df, click_count, how='left', on=['user_id', sel_fea]) 

    sorted = df.sort_values(by=['user_id', sel_fea, 'context_date'], ascending=True) # 当天数据，按点击时间升序
    first = sorted.drop_duplicates(['user_id', sel_fea])
    last = sorted.drop_duplicates(['user_id', sel_fea], keep='last')

    first['click_first'] = 1 # user_id->advertiserID, 第一次点击标志位
    first = first[['click_first']]
    df = df.join(first) # 将标志位，添加到df

    last['click_last'] = 1 # user_id->advertiserID, 最后一次点击标志位
    last = last[['click_last']]
    df = df.join(last) # 将标志位，添加到df

    df[sel_fea+'_click_cv'] = df.apply(apply_cv_trick, axis=1) # 三个点击标志位，按照特定规则，最终生成一列
    return df

def gen_cv_tricks(data,fea):
    """
    按照天数，对数据分组，然后对上下文id，每天总点击、第一次点击、最后一次点击，算转化
    """
    tricks = None
    # 遍历每一天
    for day in tqdm(set(data.clickDay)):
        df = data[data.clickDay == day] # 获取指定点击天数的数据
        df = gen_day_cv_trick(df, fea)
        day_tricks = df[['global_index', fea+'_click_cv']]
        if tricks is None:
            tricks = day_tricks
        else:
            tricks = pd.concat([tricks,day_tricks], axis=0) # 按行拼接
    data = pd.merge(data,tricks,'left','global_index') # 按列merge
    return data

#===========================================================================================

def gen_day_time_diff(df,sel_fea):
    """
    计算每天，第一次、最后一次，点击上下文id时间时间，然后对中间点击时间算diff（最终生成两个特征）
    """
    sorted = df.sort_values(by=['user_id', sel_fea, 'clickStamp'], ascending=True)  # 升序
    first = sorted.groupby(['user_id', sel_fea])['clickStamp'].first().reset_index()  # 每组取第一条记录
    first.rename(columns={'clickStamp': 'first_diff'}, inplace=True)
    last = sorted.groupby(['user_id', sel_fea])['clickStamp'].last().reset_index()  # 每组取最后一条记录
    last.rename(columns={'clickStamp': 'last_diff'}, inplace=True)

    df = pd.merge(df, first, 'left', on=['user_id', sel_fea]) # 按照用户id进行拼接
    df = pd.merge(df, last, 'left', on=['user_id', sel_fea])
    df[sel_fea+'_first_diff'] = df['clickStamp'] - df['first_diff']  # 当前点击，距离第一次点击时间差
    df[sel_fea+'_last_diff'] = df['last_diff'] - df['clickStamp']  # 当前点击，距离最后一次点击时间差
    return df

def gen_time_diff(data,fea):
    """
    按照天数，对数据分组，然后对上下文id，计算每天第一次、最后一次，
    点击上下文id时间时间，然后对中间点击时间，算diff
    """
    diff = None
    # 遍历每一天
    for day in tqdm(set(data.clickDay)):
        df = data[data.clickDay == day] # 获取指定点击天数的数据
        df = gen_day_time_diff(df, fea)
        day_time_diff = df[['global_index', fea+'_first_diff', fea+'_last_diff']]
        if diff is None:
            diff = day_time_diff
        else:
            diff = pd.concat([diff,day_time_diff], axis=0) # 按行拼接
    data = pd.merge(data,diff,'left','global_index') # 按列merge
    return data


if __name__ == '__main__':

    data = pd.read_csv(file_path + 'basic_inner_userClick.csv')
    data = data.reset_index(drop=True)  # 必须重置索引
    data['global_index'] = data.index # 全局索引

    # pandas中to_datetime格式，自带strftime函数
    tmp = pd.to_datetime(data.context_date)
    data['clickStamp'] = tmp.apply(lambda x: x.strftime('%d%H%M%S'))
    data['clickStamp'] = data['clickStamp'].astype(np.int64) # 转为int，进行加减

    # id_fea = ['user_id','item_id','shop_id','context_id','context_page_id']
    data = gen_cv_tricks(data, 'item_id')
    data = gen_time_diff(data, 'context_page_id')
    data = gen_time_diff(data, 'item_id')
    data = gen_time_diff(data, 'item_brand_id')

    # 当前特征很强（预测类别和商品类别，交叉数量特征）
    data = gen_time_diff(data, 'category_inner_flag')
    del data['category_inner_flag_first_diff']
    del data['item_brand_id_last_diff']
    del data['category_inner_flag']

    # 当前特征很强，因此采用两种bin策略，根据走势，粗细之分
    data['shop_review_positive_level_bin1'] = pd.cut(
        data['shop_review_positive_rate'], bins=[0, 0.95, 0.9999, 1], labels=False
    )
    data = gen_time_diff(data, 'shop_review_positive_level_bin1')
    del data['shop_review_positive_level_bin1']

    data['shop_review_positive_level_bin2'] = pd.cut(
        data['shop_review_positive_rate'], bins=[0, 0.95, 0.96, 0.97, 0.98, 0.99, 1], labels=False
    )
    data = gen_time_diff(data, 'shop_review_positive_level_bin2')
    del data['shop_review_positive_level_bin2']

    data.drop(['global_index', 'clickStamp'], axis=1, inplace=True)
    data.to_csv(file_path+'basic_inner_userClick_trick.csv', index=None)
