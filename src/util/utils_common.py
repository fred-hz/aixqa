from typing import List
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class SplitAndCache(object):
    def __init__(self,
                 cache_folder,
                 prefix='cache',
                 num_per_cache_file=1000):
        self.cache_folder = cache_folder
        self.cache_info_file = os.path.join(self.cache_folder, 'SUCCESS')
        self.prefix = prefix
        self.num_per_cache_file = num_per_cache_file
        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder)

    def cache_available(self):
        return os.path.exists(self.cache_info_file)

    def dump(self, obj: List[object], overwrite=False):
        if self.cache_available() and not overwrite:
            raise Exception('Cache already exists and overwrite option is False. Can not dump')
        for start in range(0, len(obj), self.num_per_cache_file):
            to_dump = obj[start: start+self.num_per_cache_file]
            pickle.dump(to_dump, open(os.path.join(self.cache_folder, f'cache_{start}'), 'wb'))
        with open(self.cache_info_file, 'w') as f:
            f.close()
        logger.info(f'finish dumping {len(obj)} objects into {self.cache_folder}')

    def load(self):
        res = []
        for file in os.listdir(self.cache_folder):
            if file.startswith('cache'):
                path = os.path.join(self.cache_folder, file)
                obj = pickle.load(open(path, 'rb'))
                res += obj
        logger.info(f'finish loading {len(res)} objects from {self.cache_folder}')
        return res
