# -*- coding: utf-8 -*-
import io
import pandas as pd
import numpy as np
import jieba
import math
from nltk.probability import FreqDist, ConditionalFreqDist
from util import *
from nltk.metrics import BigramAssocMeasures
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
dir_path = "data/save"
pos, neg = 0, 1


class Message_Classcifier(object):
    def __init__(self):
        ''
    def cut_messages(self, messages, name=None, is_load_from_file=True):
        if is_load_from_file and is_exist_file(dir_path, name):
            return load_from_pickle(dir_path, name)

        cut_messages_list = []
        stop = [line.strip() for line in io.open('data/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
        for message in messages:
            s = message.split('\n')
            fenci = jieba.cut(s[0], cut_all=False)  # False默认值：精准模式
            valid_words = list(set(fenci) - set(stop))
            cut_messages_list.append(valid_words)

        if is_load_from_file: dump_to_pickle(dir_path, name, cut_messages_list)
        return cut_messages_list

    # 获取信息量最高(前number个)的特征(卡方统计)
    def chi_features(self, number, pos_messages, neg_messages):
        pos_words = np.concatenate(pos_messages)  ##集合的集合展平成一个集合
        neg_words = np.concatenate(neg_messages)

        word_fd = FreqDist()  # 可统计所有词的词频
        cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频

        for word in pos_words:
            word_fd[word] += 1
            cond_word_fd[pos][word] += 1

        for word in neg_words:
            word_fd[word] += 1

            cond_word_fd[neg][word] += 1

        pos_word_count = cond_word_fd[pos].N()  # 积极词的数量

        neg_word_count = cond_word_fd[neg].N()  # 消极词的数量

        total_word_count = pos_word_count + neg_word_count

        word_scores = {}  # 包括了每个词和这个词的信息量

        for word, freq in word_fd.items():
            pos_score = BigramAssocMeasures.chi_sq(cond_word_fd[pos][word], (freq, pos_word_count),
                                                   total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量

            neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[neg][word], (freq, neg_word_count),
                                                   total_word_count)  # 同理

            word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

        best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                    :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的

        best_words = set([w for w, s in best_vals])
        best_features = dict([(word, True) for word in best_words])
        dump_to_pickle(dir_path, 'chi_features', best_features)
        return  best_features



    def word2vec(self, messages, tag = None, is_test_mode=False):
        vec = []
        for message in messages:
            a = {}
            for word in message:
                if word in self.selected_features.keys():
                    a[word] = 'True'
            if is_test_mode: # 测试模式，不允许贴标签
                vec.append(a)
            elif len(a) != 0: #训练模式，贴标签。如果该条短信能够提取到有用特征，则加入训练
                vec.append([a, tag])
        return vec


    def load_model(self, model_name):
        self.classifier = load_from_pickle(dir_path, model_name)
        self.selected_features = load_from_pickle(dir_path, 'chi_features')
        return self

    def fit(self, train_data, classifier, n=250, is_load_from_file=False, model_name=None, feature_name='chi_features', is_need_cut=True):
        # 保存过模型则直接加载，不需要下面的训练
        if is_load_from_file and is_exist_file(dir_path, model_name) and is_exist_file(dir_path, feature_name):
            return self.load_model(model_name)

        if is_need_cut: # 是否需要分词
            pos_messages = self.cut_messages(train_data[train_data['label'] == pos]['message'], 'train_cut_pos_messages', is_load_from_file)
            neg_messages = self.cut_messages(train_data[train_data['label'] == neg]['message'], 'train_cut_neg_messages', is_load_from_file)
        else: # 不需要分词，说明已经分词好了
            pos_messages = train_data[train_data['label'] == pos]['message'].values
            neg_messages = train_data[train_data['label'] == neg]['message'].values

        # 卡方统计返回区分度好的特征,测试的时候需要筛选
        self.selected_features = self.chi_features(n, pos_messages, neg_messages)

        # train feature
        pos_features = self.word2vec(pos_messages, pos)
        neg_features = self.word2vec(neg_messages, neg)
        train_features = np.concatenate([pos_features, neg_features])

        self.classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口
        self.classifier.train(train_features)  # 训练分类器. train_features中包含着类别
        if is_load_from_file: dump_to_pickle(dir_path, model_name, self.classifier)
        print 'train on %d instances' % (len(train_features))
        return self

    def predict(self, test_x, is_validate=False, is_need_cut=True):
        if isinstance(test_x, basestring): test_x = [test_x]
        test_messages =  self.cut_messages(test_x, is_load_from_file=False) if is_need_cut else test_x
        test_features = self.word2vec(test_messages, is_test_mode=True) # 词语转特征
        print 'test on %d instances' % (len(test_features))
        return self.classifier.classify_many(test_features)

    def acc_precision_recall_score(self, label_true, label_pred):
        print 'metric from scikit learn................'
        all_accuracy = accuracy_score(label_true, label_pred)
        pos_precision = precision_score(label_true, label_pred, pos_label=pos)
        neg_precision = precision_score(label_true, label_pred, pos_label=neg)
        pos_recall = recall_score(label_true, label_pred, pos_label=pos)
        neg_recall = recall_score(label_true, label_pred, pos_label=neg)
        pos_f1score = f1_score(label_true, label_pred, pos_label=pos)
        neg_f1score = f1_score(label_true, label_pred, pos_label=neg)

        # prints metrics to show how well the feature selection did
        print 'accuracy:', all_accuracy
        print 'pos precision:', pos_precision
        print 'pos recall:', pos_recall
        print 'neg precision:', neg_precision
        print 'neg recall:', neg_recall
        print 'pos f1 score', pos_f1score
        print 'neg f1 score', neg_f1score

        print precision_recall_fscore_support(label_true, label_pred, pos_label=pos)
        return all_accuracy, pos_precision, pos_recall, pos_f1score, neg_precision, neg_recall, neg_f1score

def cross_validate_score(data, k_fold=5):
    clf = Message_Classcifier()
    data['message'] = clf.cut_messages(data['message'], is_load_from_file=True, name='all_messages_cut') #统一切词
    data_x, data_y = data['message'], data['label']
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1)
    result = []
    for i, (train_index, validate_index) in enumerate(kf.split(data_x, data_y)):  # 会按照样本类别比例划分
        print 'fold %d start......' % i
        train_data = data.ix[train_index] # 包含标签
        validate_x = data_x.ix[validate_index]
        validate_y = data_y.ix[validate_index]
        clf.fit(train_data, LogisticRegression(class_weight={pos:0.5, neg:0.7}),n=400, is_load_from_file=False, is_need_cut=False)
        pred_y = clf.predict(validate_x, is_need_cut=False)
        result.append(clf.acc_precision_recall_score(validate_y, pred_y))
    print result
    print 'average accuracy | pos: precision, recall,f1_score | neg: precision,recall,f1_score'
    print np.mean(result, axis=0)



if __name__ == '__main__':
    # data = pd.read_csv('data/short_message.txt', names=['label', 'message'], sep='\t')
    # 交叉验证
    # cross_validate_score(data, k_fold=5)
    pass
'''
    # 划分数据 训练集:验证集 = 3 : 1, 按类别比例划分。
    pos_data = data[data['label'] == pos]
    neg_data = data[data['label'] == neg]
    posCutoff = int(math.floor(len(pos_data) * 3 / 4))
    negCutoff = int(math.floor(len(neg_data) * 3 / 4))
    train_data = pd.concat([pos_data[:posCutoff], neg_data[:negCutoff]], ignore_index=True)# 3/4训练
    validate_data = pd.concat([pos_data[posCutoff:], neg_data[negCutoff:]], ignore_index=True) # 1/4测试
    validate_x, validate_y = validate_data['message'], validate_data['label']

    # 使用逻辑回归训练
    clf = Message_Classcifier()
    print 'LogisticRegression:..........'
    clf.fit(train_data, LogisticRegression(), is_load_from_file=True, model_name='logistic_regression_model')  # 获得训练数据
    pred_y = clf.predict(validate_x, is_validate=True, is_need_cut=True)
    clf.acc_precision_recall_score(validate_y, pred_y)

    # 预测不带标签短信
    to_predict_data = pd.read_csv('data/no_label_short_message.txt', names=['message'], sep='\t')
    print "to predict data length:", len(to_predict_data)
    pred_y = clf.predict(to_predict_data['message'])
    to_predict_data['pred_label'] = pred_y
    to_predict_data.to_csv('data/no_label_short_message_pred_result.txt', sep='\t', index=False, header=None)
'''
