
#%% Necessary Packages
import tensorflow as tf

# # 
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# GPU改动
import numpy as np
import DP
from testfolder.pic import write_record 
#%% Min Max Normalizer
import datetime
NAME =  'train_' +  str(datetime.datetime.now()) + '.txt'

def MinMaxScaler(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val

#%% Start TGAN function (Input: Original data, Output: Synthetic Data)

def tgan (dataX, parameters):
    dataX_hats = list()
    # Initialization on the Graph
    tf.reset_default_graph()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])
    
    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
        
    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0
     
    # Network Parameters
    hidden_dim     = parameters['hidden_dim'] 
    num_layers     = parameters['num_layers']
    iterations     = parameters['iterations']
    batch_size     = parameters['batch_size']
    module_name    = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim          = parameters['z_dim']
    gamma          = 1
    rd_respons_p   = parameters['rd_respons_p']
    
    #%% input place holders
    
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    Z = tf.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name = "myinput_z")
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")
    
    #%% Basic RNN Cell
          
    def rnn_cell(module_name):
      # GRU
        if (module_name == 'gru'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
      # LSTM
        elif (module_name == 'lstm'):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
      # LSTM Layer Normalization
        elif (module_name == 'lstmLN'):
            rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        return rnn_cell
      
        
    #%% build a RNN embedding network      
    
    def embedder (X, T):      
      
        with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
            
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
            
            H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return H
      
    ##### Recovery
    
    def recovery (H, T):      
      
        with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
              
            r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
            
            X_tilde = tf.contrib.layers.fully_connected(r_outputs, data_dim, activation_fn=tf.nn.sigmoid) 

        return X_tilde
    
    
    
    #%% build a RNN generator network
    
    def generator (Z, T):      
      
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
            
            E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return E
      
    def supervisor (H, T):      
      
        with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
            
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers-1)])
                
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
            
            S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return S
      
      
      
    #%% builde a RNN discriminator network 
    
    def discriminator (H, T):
      
        with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
            
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
            
            # Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=tf.nn.sigmoid) 
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
            # Sigmoid交叉熵的损失函数，使得不需要激活函数。activation_fn=None
            # 均方误差下需要设置激活函数
    
        return Y_hat   
    
    
    #%% Random vector generation
    def random_generator (batch_size, z_dim, T_mb, Max_Seq_Len):
      
        Z_mb = list()
        
        for i in range(batch_size):
            
            Temp = np.zeros([Max_Seq_Len, z_dim])
            
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        
            Temp[:T_mb[i],:] = Temp_Z
            
            Z_mb.append(Temp_Z)
      
        return Z_mb
    
    #%% Functions
    
    # Embedder Networks
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    
    # Generator
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)
    
    # Synthetic data
    X_hat = recovery(H_hat, T)
    
    # Discriminator
    Y_fake = discriminator(H_hat, T)
    # 是否引入随机响应或高斯机制
    # Y_real = discriminator(H, T) 
    # 
    # DP.GaussianMechanism()
    Y_real = DP.d_rr(discriminator,H,T,rd_respons_p)
    Y_fake_e = discriminator(E_hat, T)
    
    # Variables        
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    # 改模型结构的激活函数
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # Loss for the discriminator
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    # D_loss_real =  tf.losses.mean_squared_error(tf.ones_like(Y_real), Y_real)
    # D_loss_fake = tf.losses.mean_squared_error(tf.zeros_like(Y_fake), Y_fake)
    # D_loss_fake_e = tf.losses.mean_squared_error(tf.zeros_like(Y_fake_e), Y_fake_e)
    
    # 由于交叉熵惩罚双侧（导数的系数不单调）

    # 不能开根号
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
    # Loss for the generator
    # 1. Adversarial loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    # G_loss_U = tf.losses.mean_squared_error(tf.ones_like(Y_fake), Y_fake)
    # G_loss_U_e = tf.losses.mean_squared_error(tf.ones_like(Y_fake_e), Y_fake_e)
    
    # 2. Supervised loss
    # G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,1:,:])
    G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,1:,:])
    
    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(np.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(np.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
    G_loss_V = G_loss_V1 + G_loss_V2
    
    # Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
    # 由于KL散度变为了均方误差，这一协调系数应该发生变化。
    # 由于要把D反向传播，因此不能开根号
    # G_loss =  G_loss_U +    gamma * G_loss_U_e + 0.5 *  G_loss_S + 0.5 * G_loss_V 
            
    # Loss for the embedder network
    # E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    # # # 由于要把D反向传播，因此不能开根号
    # E_loss0 =E_loss_T0
    # E_loss = E_loss0  + 0.5 * G_loss_S

    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10*tf.sqrt(E_loss_T0)
    E_loss = E_loss0  + 0.1*G_loss_S
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # optimizer原版优化器 CRR机制时使用

    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   


    # 高斯机制使用
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # compute + apply
    # sigma,Gaussion_T = DP.Cal_GaussianSigma(parameters)   #计算参数

    # temp_E0 = tf.train.AdamOptimizer().compute_gradients(E_loss0, var_list = e_vars + r_vars)
    # E0_solver = tf.train.AdamOptimizer().apply_gradients(temp_E0)
    # # E不加噪
    # temp_E = tf.train.AdamOptimizer().compute_gradients(E_loss, var_list = e_vars + r_vars)
    # temp_E = DP.Run_Gaussion(temp_E,parameters['Constant_C'],sigma)
    # E_solver = tf.train.AdamOptimizer().apply_gradients(temp_E)
    # # 混合梯度加噪
    # temp_D = tf.train.AdamOptimizer().compute_gradients(D_loss, var_list = d_vars)
    # temp_D = DP.Run_Gaussion(temp_D,parameters['Constant_C'],sigma)
    # D_solver = tf.train.AdamOptimizer().apply_gradients(temp_D)

    # # G和U不加噪
    # temp_G = tf.train.AdamOptimizer().compute_gradients(G_loss, var_list = g_vars + s_vars)     
    # G_solver = tf.train.AdamOptimizer().apply_gradients(temp_G)
    # temp_GS = tf.train.AdamOptimizer().compute_gradients(G_loss_S, var_list = g_vars + s_vars)  
    # GS_solver = tf.train.AdamOptimizer().apply_gradients(temp_GS)
    
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
   
    #%% Sessions    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #%% Embedding Learning
    
    print(str(datetime.datetime.now()) + '\n' +  'Start Embedding Network Training')
    # 编码-解码训练不够充分
    # try:
    for itt in range(iterations ):
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        # X_mb = list(dataX[i] for i in train_idx)
        # T_mb = list(dataT[i] for i in train_idx)
        # NP
        X_mb = np.array([dataX[i] for i in train_idx])
        T_mb = np.array([dataT[i] for i in train_idx])
            
        # Train embedder
        # embedder 不加噪
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        
        if itt % 100 == 0:
            print(str(datetime.datetime.now()) + '\n' + 'step: '+ str(itt) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )        
            # write_record(NAME,'step: '+ str(itt) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )
    
    # except Exception as e :
    #     print('编码器异常')
    #     print(e)
    # print(str(datetime.datetime.now()) + '\n' +  'Finish Embedding Network Training')
    
    #%% Training Supervised Loss First
    
    print(str(datetime.datetime.now()) + '\n' + 'Start Training with Supervised Loss Only')
    # try:
    for itt in range(iterations):
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)  
        # NP      
        # X_mb = np.array([dataX[i] for i in train_idx])
        # T_mb = np.array([dataT[i] for i in train_idx])
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
        # Z
        
        # Train generator  
        # generator 不加噪     
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
                        
        if itt % 100 == 0:
            print(str(datetime.datetime.now()) + '\n' + 'step: '+ str(itt) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
            # write_record(NAME,'step: '+ str(itt) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)))
                
    print(str(datetime.datetime.now()) + '\n' + 'Finish Training with Supervised Loss Only')
    # except Exception as e:
    #     print('辅助监督训练')
    #     print(e)
    
    #%% Joint Training
    
    print(str(datetime.datetime.now()) + '\n' + 'Start Joint Training')
    # Training step
    for itt in range(iterations):
    
        # Generator Training
        for kk in range(2):
        
            # Batch setting
            idx = np.random.permutation(No)
            train_idx = idx[:batch_size]     
            
            
            X_mb = list(dataX[i] for i in train_idx)
            T_mb = list(dataT[i] for i in train_idx)
            # NP      
            # X_mb = np.array([dataX[i] for i in train_idx])
            # T_mb = np.array([dataT[i] for i in train_idx])
            
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
            
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
            
            # Train embedder        
            _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
        
        #%% Discriminator Training
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
        
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)
        # NP      
        # X_mb = np.array([dataX[i] for i in train_idx])
        # T_mb = np.array([dataT[i] for i in train_idx])
        
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
            
        
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        # Train discriminator
        
        if (check_d_loss > 0.05):        
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        #%% Checkpoints
        if itt % 100 == 0:
            print(str(datetime.datetime.now()) + '\n' + 'step: '+ str(itt) + 
                ', d_loss: ' + str(np.round(step_d_loss,4)) + 
                ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
                ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
                ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
                ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
            # write_record(NAME,'step: '+ str(itt) + 
            #       ', d_loss: ' + str(np.round(step_d_loss,4)) + 
            #       ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
            #       ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
            #       ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
            #       ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4)) )
        for epsilon in parameters['epsilons']:
            e_p = DP.privacy_caculate(rd_respons_p,parameters['SubSample_Q'])
            if np.abs(itt * e_p - epsilon) <= e_p and itt*e_p <= epsilon:
                print(str(datetime.datetime.now()) + '\n' + 'step: '+ str(itt) + 
                'epsilon: ' + str(epsilon) +
                ', d_loss: ' + str(np.round(step_d_loss,4)) + 
                ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
                ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
                ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
                ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )


                # saver
                model_name = parameters['name'] + '_' + str(parameters['rd_respons_p']) + '_' + str(datetime.datetime.now())
                saver = tf.train.Saver()
                saver.save(sess,"./models/" + model_name + ".ckpt")

                
                Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
                # Z来自Z_mb，random_generator是从uniform里采样
    
                X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})    
                
                #%% List of the final outputs
                
                dataX_hat = list()
                
                for i in range(No):
                    Temp = X_hat_curr[i,:dataT[i],:]
                    dataX_hat.append(Temp)
                    
                # Renormalization
                if (Normalization_Flag == 1):
                    dataX_hat = dataX_hat * max_val
                    dataX_hat = dataX_hat + min_val

                dataX_hats.append(dataX_hat)
                print('generated data,at epsilon = ' + str(epsilon) + ' , itt = ' + str(itt))

    # except Exception as e:
    #     print('Discriminator error')
    #     print(e)   
    print(str(datetime.datetime.now()) + '\n' + 'Finish Joint Training')
    


    return dataX_hats
    #%% Final Outputs
    
    # Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
    # # Z来自Z_mb，random_generator是从uniform里采样
    
    # X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})    
    
    # #%% List of the final outputs
    
    # dataX_hat = list()
    
    # for i in range(No):
    #     Temp = X_hat_curr[i,:dataT[i],:]
    #     dataX_hat.append(Temp)
        
    # # Renormalization
    # if (Normalization_Flag == 1):
    #     dataX_hat = dataX_hat * max_val
    #     dataX_hat = dataX_hat + min_val
    
    # return dataX_hat
    