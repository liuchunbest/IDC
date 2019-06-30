# coding: utf-8
import sys
import math
import copy
import codecs
import random
from log import *
from cluster import Cluster
import numpy as np

from parameters import *
from tfidf import caculatCosine_normalize
import gc

reload(sys)
sys.setdefaultencoding('utf8')

##################################################################################
# 计算聚类个数K，计算每个词在整个特征描述集中的绝对词频
# 结果存储在word_total_count_dic中，结构为：{[word, num]}
#################################################################################
def caculateK(setences_list, all_wordlist):
    #print (u'计算聚类的个数K' + '\n')
    word_total_count_dic = {}
    num_features = len(setences_list)  # 特征的个数
    
    #计算每个词出现的总次数
    for word in all_wordlist:
        word_total_count_dic[word] = 0
    for sentence in setences_list:
        for word in sentence:
            word_total_count_dic[word]+=1
                 
    # 然后，计算聚类的个数k
    k = 0.0
    for sentence in setences_list:
        feature_word_count_dic={}
        for word in sentence:
            if word in feature_word_count_dic.keys():
                feature_word_count_dic[word]+=1
            else:
                feature_word_count_dic[word]=1
        size = len(feature_word_count_dic)
        for (word, count) in feature_word_count_dic.items():
            tmp=0.0
            if word_total_count_dic[word] > SMALLEST_WORD_FREQUENCY * num_features:  # 根据文章的内容，大于这个阈值的词才计算在内
                tmp += float(count * count) / float(word_total_count_dic[word])
        k += tmp / float(size)
    k = int(k)
    return k



#####################################################################################################
# 聚类的主函数
#####################################################################################################
def clusterByIDC(tf_idf_matrix,all_wordlist,k,DOMINATE_WORD_TFIDF):
    print "start to cluster\n"
    #writelog(u'开始聚类' + '\n')
    clusters_list = {}
    num_words_delete=0
    for i in range(0, k):
        
        #判断是否还有词存在，以防止所有的词都已经被删除了
        if num_words_delete>=len(all_wordlist):
            writelog("all the words have been deleted!!!!!!,and the clustering is stopped!!!!!")
            print "all the words have been deleted!!!!!!,and the clustering is stopped!!!!!"
            break
        
        print "the size of tf_idf_matrix is {}".format(len(tf_idf_matrix))
        print "length of all_wordslist is {}".format(len(all_wordlist))
        
        #计算各个特征之间的余弦相似度,以及特征ID与他在矩阵中的位置的对应关系
        similarity_matrix=getSimilarityMatrix(tf_idf_matrix)

        # 识别一个最优类,返回类的对象
        label = doClusterBySpectral(similarity_matrix, k)
        print "The size of label is: {}".format(len(label))
    
        cluster=returnBestCluster(tf_idf_matrix,label,k)
        cluster.setID(i)
        clusters_list[i] = cluster
    
        # 计算需要去掉的几个词，返回一个记录词ID的向量
        words_to_delete = caculateDominateWords(cluster.getCentroid(),DOMINATE_WORD_TFIDF)
        cluster_dominate_words=[]
        for item in words_to_delete:
            cluster_dominate_words.append(all_wordlist[item])
        cluster.setDominateWords(cluster_dominate_words)
        print "{} dominate words are deleted!".format(len(words_to_delete))
        print "dominate words are:"+" ".join(cluster_dominate_words)
        num_words_delete+=len(words_to_delete)
        
        #对数据集进行处理，去掉几个处于统治地位的词
        caculateNewMatrix(tf_idf_matrix,words_to_delete)
    
        print "start to compute next cluster!!\n"
        
        del similarity_matrix
        gc.collect()
        
    return clusters_list

############################################################
#计算各个特征之间的余弦相似度矩阵，并返回特征ID与他在矩阵中的位置之间的对应关系
############################################################
def getSimilarityMatrix(tf_idf_matrix):
    import numpy as np
    num_sentence=len(tf_idf_matrix)
    similarity_matrix=np.zeros((num_sentence,num_sentence))
    
    for i in range(0,num_sentence):
        for j in range(0,num_sentence):
            if i<=j:
                sim_a_b=caculatCosine_normalize(tf_idf_matrix[i], tf_idf_matrix[j])
                similarity_matrix[i][j]=sim_a_b
            else:
                similarity_matrix[i][j]=similarity_matrix[j][i]
        
    return similarity_matrix
        
    

############################################################
#按照SpectralClustering 聚类进行聚类,返回一个最优的类的成员及其知心
############################################################
def doClusterBySpectral(similarity_matrix, k):
    from sklearn.cluster import SpectralClustering
    #按照SpectralClustering进行聚类
    label_list=SpectralClustering(k,affinity='precomputed').fit_predict(similarity_matrix)
    
    return label_list
    
    
############################################################
# 对每个类，计算类的consine之和以及其均值，并返回最优的类的相关信息
############################################################
def returnBestCluster(tf_idf_matrix, label,k):
    best_cluster_id = 0
    max_consine = 0
    size=len(label)
    cluster_dic={}
    for i in range(0, size):
        if label[i] not in cluster_dic.keys():
            menmbers_list=[]
            menmbers_list.append(i)#转换各个成员的ID
            cluster_dic[label[i]]=menmbers_list
        else:
            cluster_dic[label[i]].append(i)

    #计算各个聚类的知心
    centroid_dic={}
    for i in range(0, k):
        sum_vector=[]
        for menmber_id in cluster_dic[i]:
            vectorAdd(sum_vector, tf_idf_matrix[menmber_id])
        centroid=caculateCentroid(sum_vector, len(cluster_dic[i]))
        centroid_dic[i]=centroid
        
    #计算各个聚类中的平均距离等值
    sum_consine = 0
    size_cluster = 0
    means_consine = 0
    total_consine = 0
    for i in range(0, k):
        for menmber_id in cluster_dic[i]:
            sum_consine += caculatCosine_normalize(tf_idf_matrix[menmber_id], centroid_dic[i])
        # 这里要判断下，以防止聚簇为空的情况
        #if size_cluster > 0:
        #    means_consine = sum_consine / size_cluster
        means_consine = sum_consine / len(cluster_dic[i])
        total_consine = sum_consine + means_consine
        if max_consine < total_consine:
            best_cluster_id = i
            max_consine = total_consine
            
    #返回一个cluster的ID以及其成员,知心等信息
    #这里要把各个聚簇的成员向量给记录下来，否则后面又更改了
    menmber_sentence_dic={}
    for menmber_id in cluster_dic[best_cluster_id]:
        menmber_sentence_dic[menmber_id]=copy.deepcopy(tf_idf_matrix[menmber_id])
    cluster=Cluster(centroid_dic[best_cluster_id],menmber_sentence_dic)
    return cluster
        

def vectorAdd(sum_vector, vector):
    size = len(sum_vector)
    # 如果是第一次进行一个类中的向量相加
    if size == 0:
        new_size = len(vector)
        for i in range(0, new_size):
            sum_vector.append(vector[i])    
    else:
        for i in range(0, size):
            sum_vector[i] = sum_vector[i] + vector[i]
            

def caculateCentroid(sum_vector,num):
    size = len(sum_vector)
    for i in range(0, size):
        sum_vector[i] = sum_vector[i] /num
    return sum_vector


############################################################
# 去掉最优的类的处于优势地位的词，这个阈值是0.15,并返回新的矩阵
############################################################
def caculateDominateWords(centroid_vector,DOMINATE_WORD_TFIDF):
    result = []
    size = len(list(centroid_vector))
    print "The size of centroid is: {}".format(size)
    for i in range(0, size):
        if centroid_vector[i] > DOMINATE_WORD_TFIDF:   # 这边需要找到的是j列，而不是i行
            result.append(i)
    return result


def caculateNewMatrix(tf_idf_matrix,words_to_delete):
    print "words to delete is:{}".format(words_to_delete)
    for i in range(0,len(tf_idf_matrix)):
        for j in range(0,len(tf_idf_matrix[0])):
            if j in words_to_delete:
                tf_idf_matrix[i][j]=0




##################################################################
# post-processing
##################################################################
def posProcess(clusters_list,tf_idf_matrix,SIMILARITY_TO_CENTROID,SMALLEST_SIMILARITY_FOR_MERGE):
    print "start to post processing!!\n"
    #没有被聚类的特征
    unclusted_feature_list=[]
    clusted_feature_id_list=[]
    
    print "num of clusters: ",len(clusters_list)
    
    #print "开始将每个聚簇中远离质心的特征给去掉，并重新计算聚簇质心\n"
    print "start to remove the features which is far to the centroid，and recompute the centroid\n"
    for (cluster_id, cluster) in clusters_list.items():
        #去掉离知心比较远的成员
        removeMisfits(cluster,SIMILARITY_TO_CENTROID)
        if len(cluster.getMenmberSentences())==0:
            #writelog("后处理时去掉远离质心的操作过程中的阈值设置过大，导致聚簇为空！！")
            clusters_list.pop(cluster_id)
            print "a cluster is poped!!!!!"
        else:
            #重新计算聚簇的知心
            reComputing(cluster)
            #将已经被聚类的特征ID存到一个列表中
            for menmber_id in cluster.getMenmberSentences().keys():
                if menmber_id not in clusted_feature_id_list:
                    clusted_feature_id_list.append(menmber_id)

    for i in range(0,len(tf_idf_matrix)):
        if i not in clusted_feature_id_list:
            unclusted_feature_list.append(i)
            
    print "start to find the cluster for the features which are not clustered!\n"    
    #给还没有聚类的特征寻找相应的聚簇
    reCluster(clusters_list, unclusted_feature_list,tf_idf_matrix,SIMILARITY_TO_CENTROID)

    print "star to merge similar cluster!!\n"    
    #合并相似度较大的聚簇
    mergeCluster(clusters_list,SMALLEST_SIMILARITY_FOR_MERGE)
    


####################################################################################################################
# removing misfits 对于分好的每一个类，再计算特征与质心之间的余弦相似度，将值低于0.35的特征从类中移除
####################################################################################################################
def removeMisfits(cluster,SIMILARITY_TO_CENTROID):
    #获取当前聚簇所有的句子
    menmbber_sentences_dic=cluster.getMenmberSentences()
    for (key,sentence_vector) in menmbber_sentences_dic.items():
        #print len(matrix[menmber_id])
        #print len(cluster.getCentroid())
        tmp = caculatCosine_normalize(sentence_vector, cluster.getCentroid())
        if tmp < SIMILARITY_TO_CENTROID:
            menmbber_sentences_dic.pop(key)
            writelog('***********去掉聚簇中的某些成员********'+"\n")
            writelog('menmbers of cluster: '+str(key)+'被去掉了,它与质心的相似度为('+str(tmp)+')'+'\t'+"\n")


#####################################################################################################
# 再之后是重新计算质心
#####################################################################################################
def reComputing(cluster):
    #获取当前聚簇所有的句子
    menmbber_sentences_dic=cluster.getMenmberSentences()
    num = len(menmbber_sentences_dic)
    sum_vector = []
    num=0
    for (key,sentence_vector) in menmbber_sentences_dic.items():
        vectorAdd(sum_vector, sentence_vector)
        num+=1
    sum_vector = caculateCentroid(sum_vector, num)
    #print "重新计算后的聚类质心yayayayayaya",sum_vector
    cluster.setCentroid(sum_vector)
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

#####################################################################################################
# 将没有聚过的特征进行聚类，聚类后质心向量会变吧，这里还需要重新计算聚类质心
#####################################################################################################
def reCluster(clusters_list, unclusted_feature_list,tf_idf_matrix,SIMILARITY_TO_CENTROID):
    print "The num of sentences whihc have not been clustered:", len(unclusted_feature_list)
    #writelog("没有聚类的特征个数总共有"+ str(len(unclusted_feature_list_dic))+"\n")
    
    for sentence_key in unclusted_feature_list:
        max_consine=0
        max_cluster_id=0
        for (cluster_id, cluster) in clusters_list.items():
             centroid = cluster.getCentroid()
             #计算向量与质心之间的相似度
             consine = caculatCosine_normalize(tf_idf_matrix[sentence_key], centroid)
             if consine>max_consine:
                 max_consine=consine
                 max_cluster_id=cluster_id
        if max_consine>=SIMILARITY_TO_CENTROID: # 还是没有将全部特征进行聚类,原文是0.35
            menmber_sentence_dic=clusters_list[max_cluster_id].getMenmberSentences()
            menmber_sentence_dic[sentence_key]=tf_idf_matrix[sentence_key]
            reComputing(clusters_list[max_cluster_id])
       
#####################################################################################################
# 合并聚类对象，在这里members合并后没有改变，members list变为新的特征列表
#####################################################################################################
def mergeCluster(clusters_list,SMALLEST_SIMILARITY_FOR_MERGE):
    cluster_centroid_dic={}
    centroid_id_list=[]
    for (key, cluster) in clusters_list.items():
        centroid=cluster.getCentroid()
        cluster_centroid_dic[key]=centroid
        centroid_id_list.append(key)
        
    #寻找聚簇之间相似度大于0.55的组合    
    clusters = {}
    size=len(centroid_id_list)
    for i in range(0, size):
        max_consine=0
        max_sim_cluster_id=0
        sim_pair=[]
        for j in range(i+1, size):
            centroid_1 = cluster_centroid_dic[centroid_id_list[i]]
            centroid_2 = cluster_centroid_dic[centroid_id_list[j]]
            consine = caculatCosine_normalize(centroid_1, centroid_2)
            if consine > max_consine:
                max_consine=consine
                max_sim_cluster_id=j
        if max_consine>SMALLEST_SIMILARITY_FOR_MERGE:
            print "The clusters which can be merged are:", i, max_sim_cluster_id, max_consine
            sim_pair.append(centroid_id_list[i])
            sim_pair.append(centroid_id_list[max_sim_cluster_id])
            clusters[max_consine] = sim_pair#以可以合并的两个聚簇的质心的相似度作为键值
            
    if len(clusters)==0:
        return
    
    #writelog('**********可以合并的聚簇如下************'+"\n")
    cluster_id_to_merge_list=[]
    for (consine, pair) in clusters.items():
        #writelog('如下两个聚簇可以合并:'+str(pair[0])+"\t"+str(pair[1])+'他们的相似度：'+str(consine)+"\n")
        cluster_id_to_merge_list.append(pair[0])
        cluster_id_to_merge_list.append(pair[1])
        
    cluster_id_to_merge_list=list(set(cluster_id_to_merge_list)) #去掉冗余的ID  
    while len(cluster_id_to_merge_list)>1 and len(clusters)>0:
        max_consine=max(clusters.keys())
        to_merge_pair=clusters[max_consine]
        #只要将要合并的一对任何一个已经被合并了，该合并不能进行了
        if to_merge_pair[0] not in cluster_id_to_merge_list or to_merge_pair[1] not in cluster_id_to_merge_list:
            clusters.pop(max_consine)
            continue
        #writelog("pair[0]:"+str(to_merge_pair[0]))
        #writelog("pair[1]:"+str(to_merge_pair[1]))
        #合并他们的句子
        member_sentences_a=clusters_list[to_merge_pair[0]].getMenmberSentences()
        member_sentences_b=clusters_list[to_merge_pair[1]].getMenmberSentences()
        for (key,sentence_vector) in member_sentences_b.items():
            if key in member_sentences_a.keys():#如果有相同的句子，则将他们的两个向量给合并
                vectorAdd(member_sentences_a[key], sentence_vector)
            else:
                member_sentences_a[key]=sentence_vector
        #重新计算质心
        reComputing(clusters_list[to_merge_pair[0]])
        
        #合并二者的dominate words
        dominate_words_a=clusters_list[to_merge_pair[0]].getDominateWords()
        dominate_words_b=clusters_list[to_merge_pair[1]].getDominateWords()
        for item in dominate_words_b:
            if item not in dominate_words_a:
                dominate_words_a.append(item)

        clusters_list.pop(to_merge_pair[1])#删除一个聚类，已经被合并了
        clusters.pop(max_consine)#将已经合并的聚簇组合删除

        cluster_id_to_merge_list.remove(to_merge_pair[0])#将已经聚过的聚簇的ID从这个名单中删除
        cluster_id_to_merge_list.remove(to_merge_pair[1])
        #writelog('***********合并之后的可以合并聚簇名单如下************'+"\n")
        #for item in cluster_id_to_merge_list:
        #    writelog(''+str(item)+'\t')

             
#####################################################################################################
# 聚类重命名
#####################################################################################################
def reNameCluster(sentence_list,all_words_list, cluster):
    weight = 0.0
    tag = 0
    # 从聚类的成员中选出一个描述
    centroid=cluster.getCentroid()
    menmber_sentences_dic=cluster.getMenmberSentences()
    for (key,sentence_vector) in menmber_sentences_dic.items():
        tmp = caculatCosine_normalize(sentence_vector, cluster.getCentroid())
        length = len(sentence_vector)
        count = 0.0
        num = 0
        for j in range(0, length):
            if sentence_vector[j] > 0.1:
                num += 1
                count += sentence_vector[j]
        if num > 0:
            count = count / float(num)
        #考虑两项因素的和，一个是相似度，一个是关键词的tf-idf值
        tmp+=count
        if weight <= tmp:
            weight = tmp
            tag = key
            
    #获取剩余的关键词，拼接聚簇的名字
    selected_sentence_vector=menmber_sentences_dic[tag]
    left_words_list=[]
    for j in range(0,len(all_words_list)):
        if selected_sentence_vector[j]!=0:
            left_words_list.append(all_words_list[j])
    #因为left_words_list中的词顺序可能不是原句中的词顺序，所以要调整下
    result_list=[]
    for word in sentence_list[tag]:
        if word in left_words_list:
            result_list.append(word)
        
    cluster_name = " ".join(result_list)
    cluster.setName(cluster_name)
    #writelog('the best description is: ' + str(tag) + '\t' + cluster_name) 

