# coding: utf-8
import sys
import math
import re
from log import *

reload(sys)
sys.setdefaultencoding('utf8')



###################################################
# 得到所有的词
###################################################
def getAllWordsList(all_documents_list):
    all_words_list=[]
    for document in all_documents_list:
        for word in document:
            if word not in all_words_list:
                all_words_list.append(word)
    return all_words_list


###################################################
# 得到一个文档中各个词的频率
###################################################
def computeDocumentWordTF(document_words_list):
    document_word_count_dic={}
    for word in document_words_list:
        if word in document_word_count_dic.keys():
            document_word_count_dic[word]+=1
        else:
            document_word_count_dic[word]=1
        
        
    #进行归一化处理
    total_count=0
    for (word,count) in document_word_count_dic.items():
        total_count+=document_word_count_dic[word]
    for word in document_word_count_dic.keys():
        document_word_count_dic[word]=float(document_word_count_dic[word])/float(total_count)
    return document_word_count_dic
        
        
###################################################
# 得到各个词的idf值
###################################################
def computeIDF(all_words_list,all_documents_list):
    all_word_idf_dic={}
    num_document=len(all_documents_list)
    
    for word in all_words_list:
        all_word_idf_dic[word]=0
    for word in all_words_list:
        num_supp_document=0
        for document in all_documents_list:
            if word in document:
                num_supp_document+=1
                
        word_idf = math.log(float(num_document+1) / float(num_supp_document+1), 2)
        word_idf+=1
        all_word_idf_dic[word]=word_idf
    return all_word_idf_dic




###################################################
# 得到一个文档中词的tfidf值
###################################################       
def computeDocumentWordTFIDF(document_words_list,all_word_idf_dic):
    document_word_count_dic=computeDocumentWordTF(document_words_list)
    
    document_word_tfidf_dic={}
    for (word,tf) in document_word_count_dic.items():
        document_word_tfidf_dic[word]=document_word_count_dic[word]*all_word_idf_dic[word]
        
    return document_word_tfidf_dic
    

###################################################
#将多个文档转成TF-IDF向量矩阵
################################################### 
def documentListVectorizer(all_documents_list):
    #先计算所有的词的集合
    all_words_list=getAllWordsList(all_documents_list)
    #计算每个词的idf值
    all_word_idf_dic=computeIDF(all_words_list,all_documents_list)
    #定义一个数组，将每个文档转成一个向量
    import numpy as np
    num_documents=len(all_documents_list)
    num_words=len(all_words_list)
    documents_vector_array=np.zeros((num_documents,num_words))
    for i in range(0,num_documents):
        document_word_count_dic=computeDocumentWordTF(all_documents_list[i])
        for j in range(0,num_words):
            word_tfidf=0
            word=all_words_list[j]
            if word in document_word_count_dic.keys():
                word_tfidf=document_word_count_dic[word]*all_word_idf_dic[word]
            documents_vector_array[i,j]=word_tfidf
    return documents_vector_array


##################################################################################
#计算一个选定的类别的描述中，每个词的tfidf值
##################################################################################
def computeSelectedDocumentWordTFIDF(corpus,cluster_id):
    #先计算选定类别所有的词的集合
    all_words_list=[]
    for word in corpus[cluster_id]:
        if word not in all_words_list:
            all_words_list.append(word)
            
    #计算选定类别中每个词的idf值
    all_word_idf_dic=computeIDF(all_words_list,corpus)
    #在计算选定类别文档中，每个词的tfidf值
    cluster_word_tfidf_dic=computeDocumentWordTFIDF(corpus[cluster_id],all_word_idf_dic)
   
    return cluster_word_tfidf_dic


###################################################
#在给定每个词的tfidf值得情况下，将多个文档转成TF-IDF向量矩阵
###################################################
def documentListVectorizer_normalize(all_documents_list,word_tfidf_dic):
    all_words_list=word_tfidf_dic.keys()
    #定义一个数组，将每个文档转成一个向量
    import numpy as np
    num_documents=len(all_documents_list)
    num_words=len(all_words_list)
    documents_vector_array=np.zeros((num_documents,num_words))
    for i in range(0,num_documents):
        for j in range(0,num_words):
            word_tfidf=0
            word=all_words_list[j]
            if word in all_documents_list[i]:
                word_tfidf=word_tfidf_dic[word]
            documents_vector_array[i,j]=word_tfidf
            
    #将tfidf向量归一化       
    from sklearn.preprocessing import normalize
    new_document_vector_array=normalize(documents_vector_array, norm='l2')
    
    return new_document_vector_array,num_documents,all_words_list


############################################################
# 计算两个向量之间的余弦相似度
############################################################
def caculatCosine(vector_1, vector_2):
    size=len(vector_1)
    size_2=len(vector_2)
    if size_2<=0:
        pass
    a=0
    b=0
    c=0
    for i in range(0, size):
        a+=vector_1[i]*vector_2[i]
        b+=vector_1[i]*vector_1[i]
        c+=vector_2[i]*vector_2[i]
    b=math.sqrt(b)
    c=math.sqrt(c)
    return a/(b*c)



#当两个向量归一化的情况下，返回他们的余弦相识度
def caculatCosine_normalize(vector_1, vector_2):
    size=len(vector_1)
    a=0
    for i in range(0, size):
        a+=vector_1[i]*vector_2[i]
    return a


