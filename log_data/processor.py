import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_data(data, validation_rate=0.2, test_rate=0.1, shuffle=True):
    """
    데이터를 train, validation, test 데이터로 쪼개어 csv로 저장
    :param data:
        split할 데이터. type은 pandas.DataFrame
    :param validation_rate:
        validation데이터 비율, 0~1사이의 실수여야하며 기본값은 0.2 (20%의 데이터를 validation에 사용)
    :param test_rate:
        test데이터 비율, 0~1사이의 실수여야하며 기본값은 0.1 (20%의 데이터를 test에 사용)
    :param shuffle:
        데이터를 split하기 전에 shuffle을 할 것인지 여부. 기본은 True
    :return tuple of (train_data, validation_data, test_data)
    """
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    validation_number = int(len(data) * validation_rate)
    test_number = int(len(data) * test_rate)

    train_data = data.iloc[:-(validation_number + test_number)]
    validation_data = data.iloc[-(validation_number + test_number): -test_number]
    test_data = data.iloc[-test_number:]

    return train_data, validation_data, test_data


def default_transform_log_data(log_data):
    log_data = log_data.copy(deep=True)
    log_data['pay'] = log_data['pay'].apply(lambda pay: 1 if pay > 0 else 0)

    return log_data


def normalize_feature(data, fields):
    data = data.copy(deep=True)
    for field in fields:
        min_max_scaler = MinMaxScaler()
        data[field] = min_max_scaler.fit_transform(data[field].values.reshape(-1, 1)).reshape(-1)

    return data


def one_hot_encoder(data, fields):
    data = data.copy(deep=True)
    for field in fields:
        dummies = pd.get_dummies(data[field], prefix=field, drop_first=False)
        data = pd.concat([data, dummies], axis=1)

    data.drop(fields, axis=1)
    return data
