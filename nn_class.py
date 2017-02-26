# -*- coding: utf_8 -*-  

import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

'''
1. FIR 0, IIR 1, LSTM 2
2. EEG-ECoG間でフィルター共通 0, バラバラ 1
3. 予測対象 EEG 0, ECoG 1
4. 予測の仕方 共通 0, NNをわける（フォルター特性が変わる） 1
'''

# IIR フィルター共通 EEG予測 共通　
class TimeSpacePerceptron1000(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 256
        super(TimeSpacePerceptron1000, self).__init__(
            transformECoG = L.Linear(self.ecog_num, self.eeg_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron1000, self).add_link('IIR', L.Linear(10, 1))
        #self.W = Variable(np.array([1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5]).astype('float32'))
        
    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        # EEG_t は 16 * time * time_delay
        #EEG IIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['IIR'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG IIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['IIR'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで IIR
        ecog_space = self.transformECoG(ECoG_filtered)
        ecog_space_dropout = F.dropout(ecog_space, train=train, ratio=15.0/16)
        eeg  = EEG_filtered
        return ecog_space, eeg, EEG_filtered_prev, ECoG_filtered_prev

# LSTM フィルター共通 EEG予測 共通　
class TimeSpacePerceptron2000(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 256
        super(TimeSpacePerceptron2000, self).__init__(
            transformECoG = L.Linear(self.ecog_num, self.eeg_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron2000, self).add_link('LSTM', L.LSTM(self.ecog_num, self.eeg_num))
        
    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        # EEG_t は 16 * time * time_delay
        ecog_space = self.LSTM(ECoG_t)
        ecog_space_dropout = F.dropout(ecog_space, train=train, ratio=15.0/16)
        eeg  = EEG_t
        return ecog_space, eeg, EEG_filtered_prev, ECoG_filtered_prev



# FIR フィルター共通 EEG予測 共通　
class TimeSpacePerceptron0000(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 256
        super(TimeSpacePerceptron0000, self).__init__(
            transformECoG = L.Linear(self.ecog_num, self.eeg_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0000, self).add_link('FIR', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        # EEG_t は 16 * time * time_delay
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        ecog_space = self.transformECoG(ECoG_filtered)
        ecog_space_dropout = F.dropout(ecog_space, train=train, ratio=15.0/16)
        eeg  = EEG_filtered
        return ecog_space_dropout, eeg, EEG_filtered_prev, ECoG_filtered_prev

# FIR フィルターバラバラ EEG予測 共通　
class TimeSpacePerceptron0100(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 256
        super(TimeSpacePerceptron0100, self).__init__(
            transformECoG = L.Linear(self.ecog_num, self.eeg_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0100, self).add_link('FIR_EEG', L.Linear(10, 1))
        super(TimeSpacePerceptron0100, self).add_link('FIR_ECoG', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        # EEG_t は 16 * time * time_delay
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_EEG'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_ECoG'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        ecog_space = self.transformECoG(ECoG_filtered)
        ecog_space_dropout = F.dropout(ecog_space, train=train, ratio=0.5)
        eeg  = EEG_filtered
        return ecog_space_dropout, eeg, EEG_filtered_prev, ECoG_filtered_prev
    
    
# FIR フィルター共通 ECoG予測 共通　
class TimeSpacePerceptron0010(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 256
        super(TimeSpacePerceptron0010, self).__init__(
            transformECoG = L.Linear(self.eeg_num, self.ecog_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0010, self).add_link('FIR', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        x1 = self.transformECoG(EEG_filtered)
        x1_dropout = F.dropout(x1, train=train, ratio=0.5)
        x2  = ECoG_filtered
        return x1_dropout, x2, EEG_filtered_prev, ECoG_filtered_prev
    
# FIR フィルターバラバラ ECoG予測 共通　
class TimeSpacePerceptron0110(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 256
        super(TimeSpacePerceptron0110, self).__init__(
            transformECoG = L.Linear(self.eeg_num, self.ecog_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0110, self).add_link('FIR_EEG', L.Linear(10, 1))
        super(TimeSpacePerceptron0110, self).add_link('FIR_ECoG', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_EEG'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_ECoG'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        x1 = self.transformECoG(EEG_filtered)
        x1_dropout = F.dropout(x1, train=train, ratio=0.5)
        x2  = ECoG_filtered
        return x1_dropout, x2, EEG_filtered_prev, ECoG_filtered_prev
    
# FIR フィルター共通 EEG予測 perCh
class TimeSpacePerceptron0001(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =1
        self.ecog_num = 256
        super(TimeSpacePerceptron0001, self).__init__(
            transformECoG = L.Linear(self.ecog_num, self.eeg_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0001, self).add_link('FIR', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        # EEG_t は 16 * time * time_delay
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t, pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        ecog_space = self.transformECoG(ECoG_filtered)
        ecog_space_dropout = F.dropout(ecog_space, train=train, ratio=0.5)
        eeg  = EEG_filtered
        return ecog_space_dropout, eeg, EEG_filtered_prev, ECoG_filtered_prev

# FIR フィルターバラバラ EEG予測 perCh
class TimeSpacePerceptron0101(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =1
        self.ecog_num = 256
        super(TimeSpacePerceptron0101, self).__init__(
            transformECoG = L.Linear(self.ecog_num, self.eeg_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0101, self).add_link('FIR_EEG', L.Linear(10, 1))
        super(TimeSpacePerceptron0101, self).add_link('FIR_ECoG', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        # EEG_t は 16 * time * time_delay
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t, pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_EEG'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_ECoG'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        ecog_space = self.transformECoG(ECoG_filtered)
        ecog_space_dropout = F.dropout(ecog_space, train=train, ratio=0.5)
        eeg  = EEG_filtered
        return ecog_space_dropout, eeg, EEG_filtered_prev, ECoG_filtered_prev
    
    
# FIR フィルター共通 ECoG予測 perCh
class TimeSpacePerceptron0011(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 1
        super(TimeSpacePerceptron0011, self).__init__(
            transformECoG = L.Linear(self.eeg_num, self.ecog_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0011, self).add_link('FIR', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t, pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        x1 = self.transformECoG(EEG_filtered)
        x1_dropout = F.dropout(x1, train=train, ratio=0.5)
        x2  = ECoG_filtered
        return x1_dropout, x2, EEG_filtered_prev, ECoG_filtered_prev
    
# FIR フィルターバラバラ ECoG予測 perCh　
class TimeSpacePerceptron0111(Chain):
    def __init__(self, time_range):
        self.tt = time_range
        self.eeg_num =16
        self.ecog_num = 1
        super(TimeSpacePerceptron0111, self).__init__(
            transformECoG = L.Linear(self.eeg_num, self.ecog_num)# ECoG to EEG 空間
            )
        super(TimeSpacePerceptron0111, self).add_link('FIR_EEG', L.Linear(10, 1))
        super(TimeSpacePerceptron0111, self).add_link('FIR_ECoG', L.Linear(10, 1))

    def __call__(self, EEG_t, ECoG_t, EEG_filtered_prev, ECoG_filtered_prev, train=False):
        #EEG FIR
        EEG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.eeg_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = EEG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((EEG_t[:, ch, :], pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_EEG'](x)
            EEG_filtered.append(y)
            EEG_filtered_prev[ch] = y
        EEG_filtered = F.concat(tuple(EEG_filtered))
        #ECoG FIR
        ECoG_filtered = [] #  y0, y1, y2, y3, y4 が入る ch の数だけ配列
        for ch in range(self.ecog_num):
            pre_y = [np.zeros((self.tt, 1)).astype('float32') for  i in range(5)]
            for t in range(5):
                pre_y[t][t:] = ECoG_filtered_prev[ch].data[:len(pre_y[t])-t]
                pre_y[t] = Variable(pre_y[t])
            x = F.concat((ECoG_t, pre_y[0], pre_y[1], pre_y[2], pre_y[3], pre_y[4]))
            y = self.__dict__['FIR_ECoG'](x)
            ECoG_filtered.append(y)
            ECoG_filtered_prev[ch] = y
        ECoG_filtered = F.concat(tuple(ECoG_filtered))
        # ここまで FIR
        x1 = self.transformECoG(EEG_filtered)
        x1_dropout = F.dropout(x1, train=train, ratio=0.5)
        x2  = ECoG_filtered
        return x1_dropout, x2, EEG_filtered_prev, ECoG_filtered_prev