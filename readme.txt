This project is to implement the software feature extraction approach introduced in the paper of "Negar Hariri, Carlos Castro-Herrera, Mhdi Mirakhorli, Jane Cleland-Huang, Bamshad Mobasher, 2013. Supporting domain analysis through mining and recommending features from online product listings. IEEE Transactions on Software Engineering, 39 (12):1736-1752."

Some notes about the programs are as follows:

    1.The data scraped from the website of softPedia and the results of the evaluation are in the folder of data. In detail, the filtered data from softPedia are in the folder of filted_25; the sampled data for evaluation is in the folder of selected_25_100. The results of the programs are in the folder of result. 
    2. There are five columns in the file of result. The first column is the representative sentence which IDC selected as the descriptor of the feature implied in each cluster. The second column is the sentences in the cluster. These sentences are the results after processsing. For example, the stop words are not included. The third column is the dominate words of the cluster. The fourth column is the bigram collocations which contain the dominate words. These bigram collocations are the descriptors we have used in the study.The fifth column is the represententative sentences that contain the corresponding bigram collocations.  

    3.The programs about preprocessing the data are in the file of text_process. The programs about generating overlapping sentence clusters and postprocessing these clusters are in the file of cluster_IDC. The program in the file of extract_feature is to extract bigram collocations which constain the cluster dominate words as the features.

    4.When you run the programs, the path in the file of parameters should be first modified. The path is the complete file path of the project. The programs are writed with python. Some related packages are required. 

    5.When running the program, you should first specify which categrory of products you are considering. For this purpose, you should find the parameter of dataset_id in the file of main and modify it.The value of it means the order of the selected category in the file list of selected_25_100. There are several parameters in the main file. You can also modify them. These parameters are introduced in the paper. 

    6.These programs have been writted by different persons and for a long time. So there are inconsistencies for the style of naming varables and functions. If you have interests in such programs and there are some problems, please write the email to liuchun@henu.edu.cn.

    7.We would be grateful if you find some defects about the programs and send them to us. 
