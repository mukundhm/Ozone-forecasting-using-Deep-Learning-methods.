from itertools import product
import netCDF4 as ncf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import math
import csv
from tqdm import tqdm_notebook as tqdm
from keras_tqdm import TQDMNotebookCallback
import os
import random
import tempfile
import pandas as pd
# In[2]:

a=[2,4,5,7,9,13,15,17,19,23]        #a is the hour for which we're trying to predict
b=[0,1,2,3,4,5,6,7,8,9]      # b is indexing for a

#### dates generator. 6 weeks.
no_files=365
nfiles_train=[]
nfiles_test=[]
day_count=180
for i in range(2):
    nfiles_temp=list(range(day_count,day_count+14))  #7
    nfiles_train.append(nfiles_temp)
    nfiles_temp=list(range(day_count+14,day_count+20))  #11
    nfiles_test.append(nfiles_temp)
    day_count=day_count+30
####
##  Grid reference both x-axis,y-axis
loc_y = [165,121,171]
loc_x = [67,58,39]

grid_size_x = dx = 20
grid_size_y = dy = 20

grid_sample_limit_x=[]
grid_sample_limit_y=[]
for i in loc_x:
    grid_sample_limit_x.append((i-15,i+15))
for i in loc_y:
    grid_sample_limit_y.append((i-15,i+15))

grids_x = [np.arange(x-dx, x+dx, 1) for x in loc_x]
grids_y = [np.arange(y-dy, y+dy, 1) for y in loc_y]

all_grids_x = np.concatenate(grids_x)
all_grids_y = np.concatenate(grids_y)


monitors_file = '/mnt/raid2/System/home/mmohan3/monitor_list_NC.csv'
with open(monitors_file) as csvfile:
    reader = csv.DictReader(csvfile)
    monitor_points = [(int(float(row['Col'])), int(float(row['Row'] ))) for row in reader if row['Col']] 

monitors_x, monitors_y = zip(*monitor_points)

all_grids_x=np.concatenate([all_grids_x,monitors_x])
all_grids_y=np.concatenate([all_grids_y,monitors_y])

all_points = list(product(all_grids_x, all_grids_y))

#function to get the entire file list in a directory
def get_file_list(data_dir):
    return [os.path.join(data_dir, name) for name in os.listdir(data_dir)]
#function to get index for a point based on grid reference.    
def get_index(grid,val):
    try:
        return np.where(grid==val)[0][0]
    except (TypeError,IndexError):
            if(len(val)>1):
                index_list=[]
                for i in val:
                    index_list.append(np.where(grid==i)[0][0])
                return index_list
    
short_dir='/mnt/raid2/System/home/mmohan3/weather_data/short_data_new/'

short_files = get_file_list(short_dir)
#short_files=short_files[:30]


a=[2,4,5,7,9,13,15,17,19,23]        #a is the hour for which we're trying to predict

def hours_list(l_b,f_h):
    lb_ndarr = []
    for i in a:
        tlist=list(range(i-l_b,i))           # this is where first_ndarr is created
        lb_ndarr.append([j+l_b for j in tlist])
    fh_ndarr=[] #this is the list of all indices for the future data collection
    for i in a:
        tlist=list(range(i+1,i+f_h+1))
        fh_ndarr.append(tlist)
    return lb_ndarr,fh_ndarr


b=[i for i in range(len(a))]      # b is indexing for a


# In[4]:


#@jit(nopython=True)
def dist_predictors(x,y, monitor_points,centroid=True,**kwargs):
    """
    Returns distance and angle for an arbitrary point in the domain
    w.r.t. monitor network. These are w.r.t. each monitor or centroid of all monitors
    
    Args:
        x: x_cooridate of arbitrary point
        y: y_coordinate of arbitrary point
        training_params: Training parameters containing locations of monitors
        centroid: if True, returns distance and angle from the centrod of the monitors
    
    Returns:
        Angles and distances 
    
    """
    
    
    x_centroid = np.average([point[0] for point in monitor_points])
    y_centroid = np.average([point[1] for point in monitor_points])
   
    def unit_vector(vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if centroid:   
        distance = math.sqrt((x_centroid - x)**2 + (y_centroid -y)**2)
        v1 = (x_centroid, y_centroid)
        v2 = (x,y)
        angle = angle_between(v1,v2)
        return np.array([distance, angle])
    else:
        
        distances =[math.sqrt((x - point[0])**2 + (y - point[1])**2) for point in monitor_points]
        angles = [angle_between((x,y), (point[0], point[1])) for point in monitor_points]
        da = distances + angles
        return np.array(da) 
    
#@jit(nopython=True)
def emis_predictors(x,y,tlist, d, emis_dataset, emis_species, **kwargs):
    xs = np.arange(-d,d+1) + x
    ys = np.arange(-d,d+1) + y

    num_cells = len(xs)*len(ys)
    xs_indexed=get_index(all_grids_x,xs)
    ys_indexed=get_index(all_grids_y,ys)

    grid = np.ix_(tlist,xs_indexed,ys_indexed)
    
    def get_emis_slice(emis_dataset,species):
        e1 = emis_dataset[species][:]
        e2 = e1[grid]
        return e2.reshape(len(tlist), num_cells)

    emis_seq = [ get_emis_slice(emis_dataset, species) for species in emis_species] 
    return np.concatenate(emis_seq,axis=1)    


def history_monitors_predictors(t,aq_input_data, monitor_points, **kwargs):

    monitors_x, monitors_y = zip(*monitor_points) 

    
    
    monitors_x_edit=get_index(all_grids_x,monitors_x)
    monitors_y_edit=get_index(all_grids_y,monitors_y)
  
    concs = aq_input_data['O3'][t][:,monitors_x_edit,monitors_y_edit]
    
    return concs


def met_predictors(t,monitor_points, met_dataset, met_params, **kwargs):
   
    monitors_x, monitors_y = zip(*monitor_points)     
    monitors_x_edit=get_index(all_grids_x,monitors_x)    #function to get index
    monitors_y_edit=get_index(all_grids_y,monitors_y)    #function to get index
    
    met_seq = []
    for param in met_params:
        m1 = met_dataset[param][:]
        m2 = m1[t][:,monitors_x_edit,monitors_y_edit]
        met_seq.append(m2)
    return np.concatenate(met_seq, axis=1)



def get_predictors_xyt(p,tlist,training_params,debug=False):
    """
    For each point and timestamp(s) generate a row of predictors
    """
    
    x=p[0]
    y=p[1]
    dist = dist_predictors(x,y,**training_params) #time invariant
    dist_tile = np.tile(dist,(len(tlist),1))


    emis = emis_predictors(x,y,tlist,**training_params)
    met  = met_predictors(tlist,**training_params)
    
    hist = history_monitors_predictors(tlist,**training_params)
   
    try:
        preds = np.concatenate([dist_tile, emis, hist,met], axis=1)
        return preds
    except:
        return [dist_tile,emis,hist,met]

def get_aq_input(x,y,tlist,training_params,**kwargs):
    x_indexed=get_index(all_grids_x,x)
    y_indexed=get_index(all_grids_y,y)
    dataset=training_params['aq_input_data']
    aq_input = dataset['O3'][tlist][:,x_indexed,y_indexed]
    return aq_input
# In[9]:


def train_param(inp_file_list,out_file_list):
    infile=np.load(short_dir+str(inp_file_list[0])+'.npz')
    f_input=tempfile.TemporaryFile()
    def create_npz(file_1,file_3,var):
        return np.concatenate([file_3[var],file_1[var]])
    for i in range(1,len(inp_file_list)):
        file_1=np.load(short_dir+str(inp_file_list[i])+'.npz')
        np.savez(f_input, NO=create_npz(file_1,infile,'NO'),NO2=create_npz(file_1,infile,'NO2'),PBL=create_npz(file_1,infile,'PBL')
            ,Q2=create_npz(file_1,infile,'Q2'),TEMP2=create_npz(file_1,infile,'TEMP2'),WSPD10=create_npz(file_1,infile,'WSPD10')
            ,WDIR10=create_npz(file_1,infile,'WDIR10'),O3=create_npz(file_1,infile,'O3'))
        _ = f_input.seek(0) # Only needed here to simulate closing & reopening file
        infile=np.load(f_input)
        #print(i)
    f_input = tempfile.NamedTemporaryFile(delete=True)
    outfile=np.load(short_dir+str(out_file_list[0])+'.npz')
    #outfile=outfile['O3']
    f_output=tempfile.TemporaryFile()
    for i in range(1,len(out_file_list)):
        file_1=np.load(short_dir+str(out_file_list[i])+'.npz')
        np.savez(f_output,O3=create_npz(file_1,outfile,'O3'))
        _ = f_output.seek(0) # Only needed here to simulate closing & reopening file
        outfile=np.load(f_output)
    f_output = tempfile.NamedTemporaryFile(delete=True)
    f_input.close()
    f_output.close()
    training_params = {
                   'd' : 5, #local_emissions_size
                   'emis_species': ['NO', 'NO2'], #emission_species
                   'met_params':['PBL', 'Q2', 'TEMP2', 'WSPD10', 'WDIR10'], #met parameters
                   #'met_params': ['PBL', 'WDIR10'],
                   'emis_dataset': infile,
                   'met_dataset':infile,
                   'aq_input_data':infile,
                   'o3_dataset': outfile,
                   'monitor_points':monitor_points
                 }
    return training_params
    


# In[10]:




# In[11]:  #This function is to
def limiting_points(cell_list,grid_sample_limit):
    return cell_list[np.logical_or(np.logical_and((cell_list<grid_sample_limit[0][1]),(cell_list>grid_sample_limit[0][0])),np.logical_and((cell_list<grid_sample_limit[1][1]),(cell_list>grid_sample_limit[1][0])),np.logical_and((cell_list<grid_sample_limit[2][1]),(cell_list>grid_sample_limit[2][0])))]



def sampling_function(all_points,x_no,y_no):
    x_cells=[]
    y_cells=[]
    
    x_cells=random.sample(list(all_grids_x),k=x_no)
    y_cells=random.sample(list(all_grids_y),k=y_no)  
    x_limited=limiting_points(np.array(x_cells),grid_sample_limit_x)
    y_limited=limiting_points(np.array(y_cells),grid_sample_limit_y)

    x_y=list(set(product(x_limited,y_limited)))
    return x_y


def tqdm_function(x_y_t_f,files,a,look_back,f_h):
      predictors={}
      aq_data={}
      aq_input={}
      def round_func(val):
        if(val%1>0):
          return int(val)+1
        else:
          return int(val)
      l_edge=files[round_func(look_back/24) -1]
      r_edge=files[-1*int(f_h/24)]
      for i in x_y_t_f:
          if((i[0]>l_edge) and (i[0]<r_edge)):
              inp_file_list = range(i[0]-round_func(look_back/24),i[0]+1)
              out_file_list=range(i[0],i[0]+int(f_h/24)+1)
              training_params=train_param(inp_file_list,out_file_list)#
      

              point=i[1][0]#[0]                 ## x y coordinates of the current point

              o3_dataset=training_params['o3_dataset']

              point_x=get_index(all_grids_x,int(point[0]))
              point_y=get_index(all_grids_y,int(point[1]))
              lb_ndarr,fh_ndarr=hours_list(look_back,f_h)
              print(list(inp_file_list))
              print(list(out_file_list))
              o3_dataset=o3_dataset['O3']
              aq_data[i]=o3_dataset[fh_ndarr[i[1][1]],point_x,point_y]
              print(type(aq_data[i]))
              aq_input[i]= get_aq_input(point[0],point[1],lb_ndarr[i[1][1]],training_params)
              print(i[0])
              predictors[i]=get_predictors_xyt(point,lb_ndarr[i[1][1]],training_params,debug=False).astype(int)
              print(type(predictors[i]))
      return predictors, aq_data, aq_input




def feat_ret(nfiles,a,n1,n2,look_back,f_h):
    aq_data={}

    predictors={}
    aq_input={}
    x_y=[]
    x_y=sampling_function(all_points,n1,n2)
    x_y.append(tuple([58,121]))
    x_y_t=product(x_y,b)
    x_y_t=list(x_y_t)
    
    for f_no in range(len(nfiles)):
        x_y_t_f=product(nfiles[f_no],x_y_t) 

        x_y_t_f=list(x_y_t_f)
        print(len(x_y_t_f))
        print(f_no)
        predictors_temp,aq_data_temp,aq_input_temp=tqdm_function(x_y_t_f,nfiles[f_no],a,look_back,f_h)
        predictors.update(predictors_temp)
        aq_data.update(aq_data_temp)
        aq_input.update(aq_input_temp)

    return predictors,aq_data,aq_input,x_y


look_back=30
f_h=24
a=[2,4,5,7,9,13,15,17,19,22] #the same list of hours is repeated for all the days
predictors,aq_data,aq_input,x_y=feat_ret(nfiles_train,a,35,25,look_back,f_h)
print ('\n Done getting predictors')

save_to_file = True
read_from_file = False
analyze = True
create_plots = True


def get_splits(predictors,train_fraction=0.95):
    num_predictors = len(predictors)
    pxy = list(predictors.keys())
    #print(len(aq_data))
    num_train = int(train_fraction*num_predictors)
    train_indices = np.random.choice(np.arange(num_predictors),num_train, replace=False)
    pxy_train = [pxy[i] for i in train_indices]
    predictors_train = [predictors[i] for i in pxy_train]
    aq_train=[]
    for i in pxy_train:
        aq_train.append(aq_data[i])
    #aq_train = [aq_data[i] for i in pxy_train]
    print ('number of training points', num_train, '&', len(predictors_train))

    #Random sample for Testing
    predictors_test = [predictors[i] for i in pxy if i not in pxy_train]
    aq_test = [aq_data[i] for i in pxy if i not in pxy_train]
    pxy_test = [i for i in pxy if i not in pxy_train]
    print('number of testing points', len(predictors_test), '&', len(aq_test))
    return predictors_train, aq_train, predictors_test, aq_test

if not read_from_file:
    print('Getting train and test splits')
    predictors_train, aq_train, predictors_test, aq_test = get_splits(predictors)

# In[15]:


type(predictors_train)

if not read_from_file:

    print('scaling inputs and outputs')
    from sklearn.preprocessing import MinMaxScaler

    data = np.concatenate(list(predictors.values()))
    print(len(data[1]))
    pred_scaler = MinMaxScaler()
    pred_scaler.fit(data)

    print(len(aq_data))
    for i in aq_data:
      print(len(aq_data[i]))
    aq_data_1 = np.stack(aq_data.values()).reshape(-1,24)
    aq_scaler = MinMaxScaler()
    aq_scaler.fit(aq_data_1)

    predictors_train_scaled = [pred_scaler.transform(mx) for mx in predictors_train]
    predictors_train_rnn = np.stack(predictors_train_scaled)
    aq_train_rnn = aq_scaler.transform(np.stack(aq_train).reshape(-1,24))

    predictors_test_scaled = [pred_scaler.transform(mx) for mx in predictors_test]
    predictors_test_rnn = np.stack(predictors_test_scaled)
    aq_test_rnn = aq_scaler.transform(np.stack(aq_test).reshape(-1,24))


### Save for future use.
predictors_file = '/mnt/raid2/System/home/mmohan3/predictors_data3_poster.npz'

if save_to_file == True:


    np.savez(predictors_file, predictors_train_rnn = predictors_train_rnn, #X_train, shape=(samples,tsteps,features)
                              aq_train_rnn = aq_train_rnn, #y_train
                              predictors_test_rnn = predictors_test_rnn, #X_test
                              aq_test_rnn = aq_test_rnn #y_test

            )
# In[17]:


for i in range(len(predictors_train_scaled)):
    print(len(predictors_train_scaled[i]))
# In[18]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
import copy
# In[19]:
s0,s1,s2=predictors_train_rnn.shape
print ('shape of x_train data', s0,s1,s2)
print ('shape of y_train data', aq_train_rnn.shape)


# In[20]:
    # Design Network
m = Sequential()
m.add(LSTM(128,input_shape=predictors_train_rnn.shape[1:], return_sequences=True, activation='tanh'))
m.add(LSTM(128, activation='tanh'))#
m.add(Dropout(0.50))#
m.add(Dense(128, activation='tanh'))
m.add(Dense(f_h))
m.compile(loss='mean_squared_error',optimizer='adam')


# In[21]:
predictors_train_rnn.shape[1:]


# In[22]:
print(m.summary())
print("Inputs: {}".format(m.input_shape))
print( "Outputs: {}".format(m.output_shape))


# In[23]:


history = m.fit(predictors_train_rnn, aq_train_rnn, epochs=70, 
                    validation_data=(predictors_test_rnn, aq_test_rnn),batch_size=60, verbose=2, shuffle=True
               )


# In[27]:


sns.set_style('ticks')
plt.rcParams['axes.linewidth'] = 5

fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(1,1,1)
p1 = ax.plot(history.history['loss'], linewidth=8)
p2 = ax.plot(history.history['val_loss'], linewidth=8)

ax.set_ylabel('loss (a measure of error)', fontdict={'fontsize':50})
ax.set_xlabel('No. of training epochs', fontdict={'fontsize':50})
ax.set_title('Fig 4b. Training and Validation Loss', fontdict={'fontsize':50}, y=1.08)
ax.tick_params(axis='both', labelsize=30)
ax.grid(axis='both', linewidth=3)
fig.tight_layout()
fig.show()
fig.savefig('/mnt/raid2/System/home/mmohan3/4b.pdf', format='pdf', dpi=1200)


# In[28]:


test_predict = m.predict(predictors_test_rnn)

test_predict_descaled = aq_scaler.inverse_transform(test_predict)
aq_test_rnn_descaled = aq_scaler.inverse_transform(aq_test_rnn)


# In[29]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

mae = mean_absolute_error(test_predict_descaled, aq_test_rnn_descaled)
print('MAE is {:.2} PPM'.format(mae))

mse = mean_squared_error(test_predict_descaled, aq_test_rnn_descaled)
#print('MSE is {} PPM'.format(mse))

rmse = np.sqrt(mse)
print('RMSE is {:.2} PPM'.format(rmse))

print('Percent of RMSE to Mean AQ {:.1%}'.format(rmse/aq_test_rnn_descaled.mean()))


# In[30]:


fig = plt.figure(figsize=(14,14))
plt.rcParams['axes.linewidth'] = 5
ax = fig.add_subplot(1,1,1)
p1 = ax.scatter(aq_test_rnn_descaled, test_predict_descaled, edgecolors='r', facecolor="none", s=500, linewidths=1)
ax.set_xlim([0.005,0.045])
ax.set_ylim([0.005,0.045])
ax.set_ylabel('Predicted Concentrations \n from Deep Learning Model (PPM)', fontdict={'fontsize':50}, labelpad=20)
ax.set_xlabel('Concentration from CMAQ Model\n (PPM)', fontdict={'fontsize':50}, labelpad=20)
ax.set_title('Fig 5a. Model Predictions vs \n Observed on Validation Dataset', fontdict={'fontsize':50},y=1.08)

ax.plot([0.005,0.09], [0.005,0.09], linewidth=8)
ax.tick_params(axis='both', labelsize=30)
fig.tight_layout()
fig.show()
fig.savefig('/mnt/raid2/System/home/mmohan3/4b-1.pdf', format='pdf', dpi=1200)

# In[32]:


predictors_future,aq_data_future,aq_input,x_y=feat_ret(nfiles_test,a,15,15,look_back,f_h)
print ('\n Done getting predictors')



# In[33]:


X = list(predictors_future.values())
print(len(X))
X = list(aq_data_future.values())
print(len(X))



# In[34]:


## Scale, Predict, and Descale:

def get_model_predictions(model, X_as_dict, X_scaler, y_scaler):
    
    X = list(X_as_dict.values())
    X_temp = [X_scaler.transform(mx) for mx in X]
    X_scaled = np.stack(X_temp)
    y_scaled = model.predict(X_scaled)
    return y_scaler.inverse_transform(y_scaled)
    


# In[35]:


y_model_predicted = get_model_predictions(m, predictors_future, pred_scaler, aq_scaler)
y = np.stack(aq_data_future.values()).reshape(-1,f_h)

y_labels=list(aq_data_future.keys())

y_pred_dict={}
print(y_labels)
for i in range(len(y_labels)):
    y_pred_dict[y_labels[i]]=y_model_predicted[i]

output_dir='/mnt/raid2/System/home/mmohan3/weather_data/output_data/'

day_c=[]
d=193
for i in range(2):
    day_c.append(d)
    d=d+30
b=[2,7]
loc_c=tuple([58,121])
loc_b=[]
for i in b:
    loc_b.append(tuple([loc_c,i]))

b_loc_day=list(product(day_c,loc_b))
months=['July','August','September','October']

df=pd.DataFrame()
for i in range(2):
    fig = plt.figure(figsize=(10,10))
    plt.rcParams['axes.linewidth'] = 5
    ax = fig.add_subplot(1,1,1)
    colors=['b','g']
    df=pd.DataFrame()
    
    for j in range(2):
        y_pred_trend =y_pred_dict[b_loc_day[i * 2 + j]]
        y_obs_trend =  aq_data_future[b_loc_day[i*2+j]]
        df['aq_input'+str(b_loc_day[i*2+j])]=aq_input[b_loc_day[i * 2 + j]]
        df['y_pred_trend'+str(b_loc_day[i*2+j])]=pd.Series(y_pred_dict[b_loc_day[i * 2 + j]])
        df['y_obs_trend'+str(b_loc_day[i*2+j])]=pd.Series(aq_data_future[b_loc_day[i * 2 + j]])
        
    fig.show()
    #ax.set_xlim([0.00,0.05])
    ax.set_ylim([0.00,0.09])
    ax.set_ylabel('Hr', fontdict={'fontsize':10}, labelpad=20)
    ax.set_xlabel('Concentration (ppm)', fontdict={'fontsize':10}, labelpad=20)
    ax.set_title('Fig 5b. Model Predictions vs \n Observed on Test Dataset for a 24 Hr Forecast - '+months[i], fontdict={'fontsize':15},y=1.08)
    fig.savefig('/mnt/raid2/System/home/mmohan3/6b-'+str(i)+'.pdf', format='pdf', dpi=1200)
df.to_excel(output_dir+'output_file.xlsx')
fig=plt.figure(figsize=(8,8))
plt.rcParams['axes.linewidth'] = 5
ax = fig.add_subplot(1,1,1)
rmse_list=[]
for i in range(2):
    select_points=random.sample(x_y,k=5)
    hours=[2,3,6,9]
    point_hour=list(product(select_points,hours))
    rmse=[]
    for j in point_hour:
        pred_data=y_pred_dict[tuple([day_c[i],j])]
        aq_data=aq_data_future[tuple([day_c[i],j])]
        mse=mean_squared_error(pred_data,aq_data)
        rmse.append(np.sqrt(mse))
    rmse_list.append(rmse)
plt.boxplot(rmse_list)
#plt.xticks([1,2,3,4],[months[0],months[1],months[2],months[3]])  #remove hash
plt.title('Average RMSE for the first day of JASO months sampled over 5 different coordinates', fontdict={'fontsize':10},y=1.08)
fig.savefig('/mnt/raid2/System/home/mmohan3/7b.pdf', format='pdf', dpi=1200)
