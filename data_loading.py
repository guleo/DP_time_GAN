
#%% Necessary Packages
import numpy as np
from scipy.io import arff
#%% Min Max Normalizer

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#%% Load Google Data

def DataSubSample(data,n):
    newdata = []
    i = 1
    for x in data:
        temp = 0
        
        if i < n:
            temp = temp + x
            # print(temp)
        elif i == n :
            # print('in')
            newdata.append( (temp + x)/n )
            # print(newdata)
            i = 0
        i = i + 1
        # print(f'i:{i}')
            
    return newdata
        

def testdata_loading (seq_length):

    # Load Google Data
    x = np.loadtxt('data/testdata.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX   
def google_data_loading (seq_length):

    # Load Google Data
    x = np.loadtxt('data/GOOGLE_BIG.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX_ = []
    for i in range(len(dataX)):
        outputX_.append(dataX[idx[i]])
    outputX = np.array(outputX_)
    # 输出NP数组
    return outputX
  
#%% Sine Data Generation

def sine_data_generation (No, T_No, F_No):
  
    # Initialize the output
    dataX = list()

    # Generate sine data
    for i in range(No):
      
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(F_No):              
                          
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0,0.1)            
            phase1 = np.random.uniform(0,0.1)
          
            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(T_No)] 
            Temp.append(Temp1)
        
        # Align row/column
        Temp = np.transpose(np.asarray(Temp))
        
        # Normalize to [0,1]
        Temp = (Temp + 1)*0.5
        
        dataX.append(Temp)
                
    return np.array(dataX)

def load_arff(name:str,seq_length:int):
    data, _ = arff.loadarff('./data/' + name + '_TEST.arff')
    LL = []
    for x in data: 
        L = []
        for i in range(len(x)-1):
            L.append(x[i])
        
        L = MinMaxScaler(L)
        LL.append(L[:])
    LL = np.array(LL)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(LL) - seq_length):
        _LL = LL[i:i + seq_length]
        dataX.append(_LL)
    
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    outputX_ = np.array(outputX)
    return outputX_  

def load_CSV(name:str,seq_length:int):
    # X =np.loadtxt('./data/' + name + '.csv',skiprows=1)
    # X = np.genfromtxt('./data/' + name + '.csv', delimiter=',', skip_header=1, dtype=float)
    with open('./data/' + name + '.csv', 'r') as f:
        lines = f.readlines()

    # remove double quotes and split by comma
    X = [line.replace('"', '').strip().split(',') for line in lines]

    # convert to numpy array
    arr = np.array(X)
    # arr.astype(float)
    x = arr[1:,1:].astype(float)
    dataX = []
    X = []
    for L in x:
       X.append(MinMaxScaler(L))
     # Cut data by sequence length
    for i in range(0, len(X) - seq_length):
        _x = X[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX_ = []
    for i in range(len(dataX)):
        outputX_.append(dataX[idx[i]])
    outputX = np.array(outputX_)
    return outputX

 

def load_arff_subSample(name:str,seq_length:int,subSampleF:int):
    data, _ = arff.loadarff('./data/' + name + '_TEST.arff')
    LL = []
    for x in data: 
        L = []
        for i in range(len(x)-1):
            L.append(x[i])
        L = DataSubSample(L,subSampleF)
        L = MinMaxScaler(L)
        LL.append(L[:])
    LL = np.array(LL)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(LL) - seq_length):
        _LL = LL[i:i + seq_length]
        dataX.append(_LL)
    
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    outputX_ = np.array(outputX)
    return outputX_  


        
def save_data(name,data):
    a = np.array(data)
    np.save('./NewData/' + name + '.npy',a)
    print(name + 'is saved.')

def load_new_data(name):
    a = np.load('./NewData/' + name + '.npy')
    a = a.tolist()
    print(name + "is read.")
    return a


if __name__ == '__main__':
   X =  load_CSV('energy',24)
   print(X.shape)
   print(X[0])
