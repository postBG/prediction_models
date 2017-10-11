import os
import pandas as pd

batch_ids = ['20171001', '20171002', '20171003', '20171004', '20171005', '20171006', '20171007', '20171008']


class LogDataLoader:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir

    def load_batch(self, batch_number=1):
        """
        log_dir 아래에 있는 log 데이터를 읽어 pandas dataframe으로 만든다.

        :param batch_number: [1, 8] 범위의 숫자, 기본값은 1
        :return: 해당 batch를 읽어 pandas의 dataframe을 반환
        """
        batch_id = '2017100' + str(batch_number)
        if batch_id not in batch_ids:
            raise ValueError("batch_number는 1~8 중 하나의 자연수여야 합니다.")

        return pd.read_csv(os.path.join(self.log_dir, 'convertlog.' + batch_id), sep='\t')
