from tqdm import tqdm
import pandas as pd

file_path = 'C:\\CTR\\data\\feature\\'

def gen_his_day_cvr_smooth(data, day, fea, alpha=0.35): # 调参 0.25->0.35
    dfCvr = data[data.clickDay < day]
    dfCvr = pd.get_dummies(dfCvr, columns=['is_trade'], prefix='label')
    dfCvr = dfCvr.groupby([fea], as_index=False).sum()
    dfCvr[fea+'_cvr'] = (dfCvr['label_1']+alpha) / (dfCvr['label_0']+dfCvr['label_1']+alpha*2) # 简单平滑策略
    ret = pd.merge(
        data.loc[data.clickDay==day,['clickDay',fea]], dfCvr[[fea,fea+'_cvr']],'left',on=[fea]
    )
    ret.drop_duplicates(['clickDay', fea], inplace=True)
    ret.sort_values(['clickDay', fea], inplace=True)
    return ret[['clickDay', fea, fea+'_cvr']]

def gen_his_cvr_smooth(data, fea):
    his_cvr_smooth = None
    for day in tqdm(range(19,26)): # 遍历每一天 
        day_cvr_smooth = gen_his_day_cvr_smooth(data, day, fea)
        if his_cvr_smooth is None:
            his_cvr_smooth = day_cvr_smooth
        else:
            his_cvr_smooth = pd.concat([his_cvr_smooth, day_cvr_smooth],axis=0)
    # his_cvr_smooth.fillna(0, inplace=True) NaN填充之后，效果反而变差
    data = pd.merge(data,his_cvr_smooth,'left',['clickDay',fea])
    return data

if __name__ == '__main__':
    data = pd.read_csv(file_path + 'basic_inner_userClick_trick.csv')
    data = data.reset_index(drop=True) # 必须重置索引
    data = gen_his_cvr_smooth(data,'user_id')
    data = gen_his_cvr_smooth(data,'item_id')
    data = gen_his_cvr_smooth(data,'item_sales_level')
    data = gen_his_cvr_smooth(data,'context_page_id')
    data.to_csv(file_path + 'basic_inner_userClick_trick_cvr.csv', index=None)







