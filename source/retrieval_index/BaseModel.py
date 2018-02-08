# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 拟计划统一模型文件的编写

from abc import ABCMeta, abstractmethod


class BaseModel:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def build_model(self):
        raise NotImplementedError
