# -*- coding: utf-8 -*-

"""
    expert_generator.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""


class ExpertGenerator(object):

    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def update_memory(self, **kwargs):
        pass

    def iter_call_back(self, **kwargs):
        pass

    def reset_idx(self, idx=0):
        self.current_idx = idx

    def gen_sample_data(self, **kwargs):
        raise NotImplementedError("Must override gen_sample_data")
