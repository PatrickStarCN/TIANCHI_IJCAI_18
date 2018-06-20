import pandas as pd

raw_data_path = 'C:\\CTR\\data\\'
feature_data_path ='C:\\CTR\\data\\feature\\'

def load_csv(path):
    return pd.read_csv(path)

def write_csv(obj, path):
    obj.to_csv(path, index=None)

def addTime(data):
    """
    对点击时间，抽取天、小时特征
    """
    data['context_date'] = pd.to_datetime(data['context_date'], format='%Y-%m-%d %H:%M:%S')
    data['clickDay'] = data['context_date'].apply(lambda x:x.day)
    data['clickHour'] = data['context_date'].apply(lambda x:x.hour)
    return data

def gen_user_level_click(data):
    """
    统计每个用户星级，历史点击记录数量
    """
    user_level_day_click = pd.DataFrame(
        data.groupby(['user_star_level']).size()
    ).reset_index().rename(columns={0:'user_level_click_day'})
    data = pd.merge(data,user_level_day_click,'left',['user_star_level'])
    return data

def gen_user_day_click(data):
    """
    统计每个user，每一天，点击记录数量
    """
    user_day_click = pd.DataFrame(
        data.groupby(['user_id','clickDay']).size()
    ).reset_index().rename(columns={0:'user_click_day'})
    data = pd.merge(data,user_day_click,'left',['user_id','clickDay'])
    return data

if __name__ == '__main__':
    data=load_csv(feature_data_path+'basic_inner.csv')
    data=addTime(data)
    data=gen_user_level_click(data)

    data=gen_user_day_click(data)
    write_csv(data,feature_data_path+'basic_inner_userClick.csv')

