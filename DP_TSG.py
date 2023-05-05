import numpy as np
import sys
import argparse
import DP_Gaussion_TGAN
import DP_tgan
import tgan

from data_loading import google_data_loading, sine_data_generation,load_arff,save_data,load_arff_subSample,load_CSV
from DP import CRR_Itt


sys.path.append('metrics')

from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics

# from testfolder.pic import write_record

import datetime

record_name = str(datetime.datetime.now()) + '.txt'

print('Finish importing necessary packages and functions')
# write_record(record_name,'Finish importing necessary packages and functions')

# pre
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = "ECG200")
parser.add_argument('--rd_p', type=float, default = 0.55)
parser.add_argument('--itt', type=int, default = 800)
parser.add_argument('--batchsize', type=int, default = 128)
parser.add_argument('--privacy', type=bool, default = True)
parser.add_argument('--module_name', type=str, default = 'gru')
parser.add_argument('--hidden_dim', type=int, default = 4)
parser.add_argument('--Mech', type=str, default = 'CRR')
 
# dataset name：  ECGFiveDays CBF SonyAIBORobotSurface2 数据集数量小于1000不行
#  ItalyPowerDemand    FaceAll FacesUCR TwoLeadECG MoteStrain
#  FaceAll InsectWingbeatSound ECG5000 ChlorineConcentration TwoPatterns
args = parser.parse_args()
# Data
# data_set = ['google','sine']
# data_name = data_set[0]
data_name = args.dataset
# Experiments iterations
Iteration = 10
Sub_Iteration = 10

# Data Loading
seq_length = 24

if data_name == 'google':
    dataX = google_data_loading(seq_length)
elif data_name == 'sine':
    No = 10000
    F_No = 5
    dataX = sine_data_generation(No, seq_length, F_No)
elif data_name == 'energy':
    dataX = load_CSV('energy',seq_length)
else:
    dataX = load_arff(data_name,seq_length)
    # 下采样数据
    
    if dataX.shape[2] >= 40:
        n = dataX.shape[2]//40
        print(f"数据长度 {dataX.shape[2]},下采样到{n+1}分之一")
        dataX = load_arff_subSample(data_name,seq_length,n+1)
    
    # dataX = load_arff_subSample(data_name,seq_length,2)


# write_record(record_name,data_name + ' dataset is ready.')
print(data_name + ' dataset is ready.')
print(f'dataX.shape:{dataX.shape}')


# Newtork Parameters
parameters = dict()

parameters['Mech'] = args.Mech
parameters['hidden_dim'] = len(dataX[0][0,:]) * args.hidden_dim
parameters['num_layers'] = 3
parameters['batch_size'] = args.batchsize
parameters['module_name'] = args.module_name   # Other options: 'lstm' or 'lstmLN'
parameters['z_dim'] = len(dataX[0][0,:]) 

parameters['rd_respons_p'] = args.rd_p     #随机响应参数 200轮 epsilon=8   随机响应参数 800轮 epsilon=8
# parameters['epsilons'] = [2.0,4.0,8.0]
parameters['epsilons'] = [0.1,0.25,0.5,1.0,2.0,4.0,8.0]
parameters['privacy'] = args.privacy
parameters['name'] = data_name
parameters['SubSample_Q'] = dataX.shape[1]/parameters['batch_size']
parameters['GaussianItt'] = [400,800,1600]
parameters['Constant_C'] = 4
parameters['iterations'] = CRR_Itt(parameters['epsilons'][-1],parameters['rd_respons_p'],parameters['SubSample_Q'])

print('Parameters are ' + str(parameters))
# write_record(record_name,'Parameters are ' + str(parameters))

# Experiments
# Output Initialization
Discriminative_Score = list()
Predictive_Score = list()



print('Start iterations') 
# write_record(record_name,'Start iterations')    
# Each Iteration
for it in range(Iteration):

    
    # Synthetic Data Generation
    if parameters['Mech'] == "CRR":
        dataX_hats = DP_tgan.tgan(dataX, parameters)   
    elif parameters['Mech'] == 'Gaussian':
        dataX_hats = DP_Gaussion_TGAN.tgan(dataX, parameters)   
    elif parameters['Mech'] == 'None':
        dataX_hats = tgan.tgan(dataX, parameters) 
    else:
        print('选择恰当的机制')
      
    print('Finish Synthetic Data Generation')
    save_data('New_' + data_name +'_'+str(parameters['rd_respons_p'])+'_'+ str(datetime.datetime.now()) ,dataX_hats)
    
    # write_record(record_name,'Finish Synthetic Data Generation')    
    # Performance Metrics
    i = 0
    DisList = []
    PreList = []
    for dataX_hat in dataX_hats:

    #计算多个数据集 
        
        print(str(datetime.datetime.now()))
        print("正在评估第" + str(i+1) +"个数据集，共"+str(len(dataX_hats))+"个")

        # 先画图
        pic_name =  parameters['Mech'] + "_" + data_name + "_" +"eps_"+ str(parameters['epsilons'][i]) +"_" +str(it) +  "_"+ str(parameters['rd_respons_p'])    
        
        PCA_Analysis(dataX, dataX_hat,pic_name)
        tSNE_Analysis(dataX, dataX_hat,pic_name)   
        


         # 1. Discriminative Score

        Acc = list()
        for tt in range(Sub_Iteration):
            Temp_Disc = discriminative_score_metrics (dataX, dataX_hat)
            Acc.append(Temp_Disc)
        
        Discriminative_Score.append(np.mean(Acc))
        print(str(datetime.datetime.now()))
        DisList.append(np.round(np.mean(Discriminative_Score),4))
        print('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))
        
        # 2. Predictive Performance
        MAE_All = list()
        for tt in range(Sub_Iteration):
            MAE_All.append(predictive_score_metrics (dataX, dataX_hat))
            
        Predictive_Score.append(np.mean(MAE_All)) 

       
        print(str(datetime.datetime.now()))
        
        print('Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))
        PreList.append((np.round(np.mean(Predictive_Score),4)))
        i = i + 1
    print(f'Mech:{parameters["Mech"]},itt:{it};汇总Dis:{DisList}')
    print(f'Mech:{parameters["Mech"]},itt:{it};汇总Pre:{PreList}')
    
print('Finish TGAN iterations')
# write_record(record_name,'Finish TGAN iterations')    




# write_record(record_name,'Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))    
# write_record(record_name,'Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4))) 