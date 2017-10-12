import os
import pandas as pd


class LogDataLoader:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir

    def load_batch(self, batch_number=1):
        """
        log_dir 아래에 있는 log 데이터를 읽어 pandas dataframe으로 만든다.

        :param batch_number: 1이상의 자연수. 기본값은 1
        :return: 해당 batch를 읽어 pandas의 dataframe을 반환
        """
        batch_id = '201710' + str(batch_number).zfill(2)
        return pd.read_csv(os.path.join(self.log_dir, 'convertlog.' + batch_id), sep='\t')
