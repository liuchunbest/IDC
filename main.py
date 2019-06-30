# coding: utf-8
import sys

from log import *
#from numpy import *
import codecs
import copy
from datetime import datetime
from parameters import *
reload(sys)
sys.setdefaultencoding('utf8')


writelog("**********程序开始运行****************")

####################################################################
# 按照IDC的方式进行聚类
####################################################################

from cluster_IDC import *
from cluster import Cluster
import numpy as np
import gc

from extract_feature import *

    
def IDC_algorithm(k_number_features,tfidf_matrixt, DOMINATE_WORD_TFIDF,SIMILARITY_TO_CENTROID,SMALLEST_SIMILARITY_FOR_MERGE):
    #迭代聚类
    
    #聚类
    cluster_dic=clusterByIDC(tfidf_matrixt,all_words_list,k_number_features,DOMINATE_WORD_TFIDF)
    #输出聚类结果
    writelog("The result of IDC*****************")
    output(cluster_dic)
    
    #后处理
    posProcess(cluster_dic,tfidf_matrixt,SIMILARITY_TO_CENTROID,SMALLEST_SIMILARITY_FOR_MERGE)

    k=len(cluster_dic)#重新计算聚簇的个数，以防止后处理过程给变化了
    print "The finall num of clusters is: {}".format(k)
    writelog("The finall num of clusters is: {}".format(k))
    
    #对聚簇进行重命名
    for (key, cluster) in cluster_dic.items():
        reNameCluster(setences_list,all_words_list,cluster)
    
    #输出最终聚类结果
    writelog("The finall result of clustering*****************")
    file_name="antivirus_"+str(k_number_features)+","+str(DOMINATE_WORD_TFIDF)+","+str(SIMILARITY_TO_CENTROID)+","+str(SMALLEST_SIMILARITY_FOR_MERGE)
    #output(cluster_dic)
    save_result(cluster_dic,file_name)



def output(cluster_dic):
    #输出每个聚簇的结果
    size=len(all_words_list)
    for (key,cluster) in cluster_dic.items():
        writelog("************"+str(key)+"*****************")
        writelog("The sentences in the cluster:**** ")
        menmber_sentences_dic=cluster.getMenmberSentences()
        for (sentence_key,sentence_vector) in menmber_sentences_dic.items():
            left_sentence_words=[]
            for i in range(0,size):
                if sentence_vector[i]!=0:
                    left_sentence_words.append(all_words_list[i])
            #按照原句子中词的顺序输出约简之后的句子，有些词可能被删掉了
            result_words_list=[]
            for word in setences_list[sentence_key]:
                if word in left_sentence_words:
                    result_words_list.append(word)
            writelog(str(sentence_key)+": "+" ".join(result_words_list))
        writelog("The original sentences in the cluster:*** ")
        for (sentence_key,sentence_vector) in menmber_sentences_dic.items():
            writelog(str(sentence_key)+": "+original_sentences_list[sentence_key])
        writelog("The clusterdonminate words are:*** ")
        writelog(" ".join(cluster.getDominateWords()))
        writelog("The cluster name is:*** ")
        writelog(cluster.getName())
        writelog("*******************end*****************")


#将结果输出到
def save_result(cluster_dic,file_name):
    import xlwt
    wbk = xlwt.Workbook()
    cluster_sheet = wbk.add_sheet('sheet')
    name_writed=False
    k=0
    row=0
    
    size=len(all_words_list)
    for (key,cluster) in cluster_dic.items():
        name_writed=False
        if name_writed==False:
            #写入名字
            cluster_sheet.write(row,0,cluster.getName())#写入特征名字
            name_writed=True
            
        #写入社区的句子
        menmber_sentences_dic=cluster.getMenmberSentences()
        #实验之用
        dominate_words_list=cluster.getDominateWords()
        key_reduced_sentences_dic={}
        
        for (sentence_key,sentence_vector) in menmber_sentences_dic.items():
            left_sentence_words=[]
            for i in range(0,size):
                if sentence_vector[i]!=0:
                    left_sentence_words.append(all_words_list[i])
            #按照原句子中词的顺序输出约简之后的句子，有些词可能被删掉了
            result_words_list=[]
            for word in setences_list[sentence_key]:
                if word in left_sentence_words:
                    result_words_list.append(word)
            #实验之用
            key_reduced_sentences_dic[sentence_key]=result_words_list
            
        #这个地方产生长度为2的词组以作为特征，仅做实验之用
        selected_feature_list,selected_sentences_list=extractFeatures(key_reduced_sentences_dic.values(),dominate_words_list)
        dominate_words_writed=False
        num_features=0
        for (sentence_key, reduced_sentence) in key_reduced_sentences_dic.items():
            #写入社区中的句子        
            cluster_sheet.write(row,1," ".join(reduced_sentence))
            #写入社区中的句子的原有完整句子
            cluster_sheet.write(row,2,original_sentences_list[sentence_key])
            
            if dominate_words_writed==False:
                #写入dominate words
                cluster_sheet.write(row,3," ".join(dominate_words_list))
                dominate_words_writed=True

            #写入抽取的特征
            if selected_feature_list!=None and num_features<len(selected_feature_list):
                cluster_sheet.write(row,4," ".join(list(selected_feature_list[num_features])))
                #写入社区中的句子的原有完整句子
                cluster_sheet.write(row,5," ".join(selected_sentences_list[num_features]))
                num_features+=1
            row+=1
            
        if selected_feature_list!=None and num_features<len(selected_feature_list):
            t=num_features
            for i in range(t, len(selected_feature_list)):
                cluster_sheet.write(row,4," ".join(list(selected_feature_list[num_features])))
                #写入社区中的句子的原有完整句子
                cluster_sheet.write(row,5," ".join(selected_sentences_list[num_features]))
                num_features+=1
                row+=1
     
    final_file_name=path+"data\\result\\"+file_name+".xls"        
    wbk.save(final_file_name)
            
    
#########################################################################################
# main
#########################################################################################
if __name__=="__main__":
    #获取当前时间
    start_time=datetime.now()

    from text_process import *
    print "star to get data"
    corpus,cluster_sentences_list,cluster_original_sentences_list=getALLFileContent(path+"data\\selected_25_100")
    
    #antivirus的标号为0,compress的ID为3
    dataset_id=0
    print "there are {} sentences".format(len(cluster_sentences_list[dataset_id]))
    writelog("there are {} sentences".format(len(cluster_sentences_list[dataset_id])))
    
    print "start to get tfidf data"
    writelog("star to get tfidf data")
    from tfidf import computeSelectedDocumentWordTFIDF,documentListVectorizer_normalize
    #计算当前类的每个词的idf值
    cluster_word_tfidf_dic=computeSelectedDocumentWordTFIDF(corpus,dataset_id)
    #计算当前类的tfidf矩阵
    tf_idf_matrix,num_sentence,all_words_list= documentListVectorizer_normalize(cluster_sentences_list[dataset_id],cluster_word_tfidf_dic)

 
    
    #选定类别数据的所有句子列表
    setences_list=cluster_sentences_list[dataset_id]
    original_sentences_list=cluster_original_sentences_list[dataset_id]
    
    # 计算聚类的个数k
    print "start to compute the number of clusters: K\n"
    writelog("start to compute the number of clusters: K")
    #k = caculateK(setences_list, all_words_list)

    #k=20

    #writelog("num of clusters by IDC is :" + str(k))
    #print "num of clusters by IDC  is :" + str(k)+"\n"
    
    #针对上述四个变量，改变这些变量的值，不断执行聚类过程，寻找使得纯度和互信息最大的组合
    #a=DOMINATE_WORD_TFIDF
    #b=SIMILARITY_TO_CENTROID
    #b=SMALLEST_SIMILARITY
    #c=SMALLEST_SIMILARITY_FOR_MERGE
    k_array=[20]
    a_array=np.arange(0.1, 1, 1)#这次取0.1
    b_array=np.arange(0.1, 1, 1)
    c_array=np.arange(0.1,1,0.1)#这次取0.5

    a_list=a_array.tolist()
    b_list=b_array.tolist()
    c_list=c_array.tolist()


    for k in k_array:
        for a in a_list:
            for b in b_list:
                for c in c_list:
                    writelog("*********************************A NEW RUN OF IDC*********************************")
                    #复制一份数据集，因为在迭代的过程中该数据集会发生变化
                    copy_tf_idf_matrix=copy.deepcopy(tf_idf_matrix)
    
                    writelog("parameters are:")
                    writelog("DOMINATE_WORD_TFIDF is: {}".format(a))
                    writelog("SIMILARITY_TO_CENTROID is: {}".format(b))
                    writelog("SMALLEST_SIMILARITY_FOR_MERGE is: {}".format(c))
                
                    IDC_algorithm(k,copy_tf_idf_matrix,a,b,c)
                    writelog("*********************************END RUN OF IDC*********************************")

                    del copy_tf_idf_matrix
                    gc.collect()


    #IDC_algorithm(k,0.15,0.35,0.55)
    #获取当前时间
    ent_time=datetime.now()
    cost_time=(ent_time-start_time).seconds
    print "****************IDC total time {} seconds***************".format(cost_time)
    writelog("****************IDC 10 times: total time {} seconds***************".format(cost_time))

