import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import joblib

########### DIR PATH ###########

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')
#SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

########### LOAD MODEL AND DATA ###########

# model_class = joblib.load(f'{MODEL_DIR}\lstm_class_streamlit.joblib')
# model_pred = load_model(f'{MODEL_DIR}\lstm_second_it.h5')
# scaler = joblib.load(f'{MODEL_DIR}\scaler.joblib')

########### Functions ###########

### REGRESSION PART ###

# Define custom metric for model evaluation
def r2_keras(y_true, y_pred):
    """
    Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used.
    This means for testing we need to drop those which are below the window-length.
    An alternative would be to pad sequences so that we can use shorter ones """
    
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :] 

@st.cache
def make_pred_prediction(seq_array_last, label_array_last):
    # y_pred = (model_pred.predict(seq_array_last) > 0.5).astype("int32")
    # y_true = label_array_last
    y_pred = (model_pred.predict(seq_array_last, verbose=1, batch_size=32)).astype("int32")
    y_true = label_array_last.astype("int32")
    return y_pred, y_true    

@st.cache
def make_class_prediction(seq_array_last, label_array_last):
    y_pred = (model_class.predict(seq_array_last) > 0.5).astype("int32")
    y_true = label_array_last
    return y_pred, y_true

@st.cache(allow_output_mutation=True)
def load_models():
    model_class = joblib.load(f'{MODEL_DIR}\lstm_class_streamlit.joblib')
    model_pred = load_model(f'{MODEL_DIR}\lstm_streamlit_test.h5', custom_objects={"r2_keras": r2_keras})
    scaler = joblib.load(f'{MODEL_DIR}\scaler.joblib')
    return model_class, model_pred, scaler

def generate_label_columns(df, first_window=15, second_window=30):
    # Create 2 labels with each different window
    df['label1'] = np.where(df['RUL'] <= second_window, 1, 0 )
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= first_window, 'label2'] = 2

    return df

def normalizing_data(df,scaler):
    # Create new column for normalizing cycle
    df['cycle_norm'] = df['cycle']

    # Columns to normalize
    cols_normalize = df.columns.difference(['id','cycle','RUL','label1','label2'])

    # Create a df with normalized values
    norm_df = pd.DataFrame(scaler.transform(df[cols_normalize]), 
                           columns=cols_normalize, 
                           index=df.index)

    # Create new df with normalize values and add it to the initial test_df
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    df = df.reset_index(drop=True)
    return df

def get_last_sequence(df, sequence_length, sequence_cols):
    # Get last sequence of each id in the test data
    seq_array_last = [df[df['id']==id][sequence_cols].values[-sequence_length:] 
                           for id in df['id'].unique()
                           if len(df[df['id']==id]) >= sequence_length]

    seq_array_last = np.asarray(seq_array_last).astype(np.float32)

    # Using this y_mask in order to keep only engines with at least 50 cycles
    y_mask = [len(df[df['id'] == id]) >= sequence_length for id in df['id'].unique()]

    return seq_array_last, y_mask

def get_label(df, y_mask):
    label_array_last = df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_last = label_array_last.reshape(label_array_last.shape[0],1).astype(np.float32)

    return label_array_last

def plot_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d')
    return ax

def sensor_visualizer(df, sensors_list, engine_id):
    # Visualizing different sensors
    values = df[df.id == engine_id].values
    groups = sensors_list
    i = 1
    #fig = plt.figure(figsize=(10,20))
    fig = plt.figure(figsize=(10,25))
    plt.subplots_adjust(hspace=0.80)
    #fig.tight_layout(h_pad=5.0)
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(f'Sensor {sensors_list[i - 1]}')
        plt.xlabel('Cycles') 
        plt.ylabel('Values') 
        #plt.title(df.columns[group], y=0.5, loc='right')
        i += 1 
    st.pyplot(fig)

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    return df


########### Global Variables ###########

### Display only remaining engines after y_mask ###
# CAUTION : NEED TO ADAPT IT TO NUMBER OF ENGINES TO PREDICT (100 here, but more in other test dataset)
engine_remaining = []
engine_to_remove = [1, 2, 14, 22, 25, 39, 85]
for i in range(1,101):
    if i not in engine_to_remove:
        engine_remaining.append(i)

### Load Models ###
model_class, model_pred, scaler = load_models()

########### Sidebar ###########
with st.sidebar:
    # File Uploader
    engines_data_file = st.file_uploader("Upload your file", type="csv", key='file_uploader')
    # Prediction Type
    prediction_type = st.radio(
        'Choose between these prediction types',
        ('Classification (failure in 15 cycles)', 'Regression (cycles n° until failure)')
    )

########### Title Intro ###########
st.title('SmartClass')

########### Regression Part ###########
if prediction_type == 'Regression (cycles n° until failure)':
    if engines_data_file is not None:
        st.success('File uploaded!')
        df = load_data(engines_data_file)
        #st.dataframe(df)

        ### TimeSeries prep ###
        # Define sequence length
        sequence_length = 50

        # Pick the feature columns 
        sensor_cols = ['s' + str(i) for i in range(1,22)]
        sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
        sequence_cols.extend(sensor_cols)

        # We pick the last sequence for each id in the test data
        seq_array_test_last = [df[df['id']==id][sequence_cols].values[-sequence_length:] 
                            for id in df['id'].unique() if len(df[df['id']==id]) >= sequence_length]
        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

        # Similarly, we pick the labels
        y_mask = [len(df[df['id']==id]) >= sequence_length for id in df['id'].unique()]
        label_array_test_last = df.groupby('id')['RUL'].nth(-1)[y_mask].values
        label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

        y_pred_test = model_pred.predict(seq_array_test_last).astype(np.int32)
        y_true_test = label_array_test_last

        # Create dataframe with real and predicted values for each engine
        pred_rul_df = pd.DataFrame({'engine_id': engine_remaining, 'real': y_true_test[:,0], 'prediction':  y_pred_test[:,0]})
        # st.dataframe(pred_rul_df)

        ### 15/30 cycles prediction part ###
        st.header("Engine")
        engine_number = st.selectbox('Choose an engine', engine_remaining)
        if engine_number:
            value = pred_rul_df[pred_rul_df['engine_id'] == engine_number]['prediction']

        col1, col2, col3 = st.columns([3,2,3])
            
        with col2:
            st.header(f"Engine {engine_number}")
            st.metric(label="RUL", value=value) 

        ### Collapsed parts ###
        with st.expander(f"See difference between prediction and real RUL"):
            fig_verify = plt.figure(figsize=(10, 5))
            plt.plot(y_pred_test, color="blue")
            plt.plot(y_true_test, color="green")
            plt.title('Remaining Useful Life Prediction')
            plt.ylabel('RUL')
            plt.xlabel('Engine')
            plt.legend(['predicted', 'real'], loc='upper left')
            plt.show()
            st.pyplot(fig_verify)

        with st.expander(f"Evaluation metrics for the actual model"):
            scores_test = model_pred.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
            st.write('\nMAE: {}'.format(scores_test[1]))
            st.write('\nR^2: {}'.format(scores_test[2]))

########### Classification Part ###########
else:
    if engines_data_file is not None:
        st.success('File uploaded!')
        df = load_data(engines_data_file)

        ### Preprocessing ###
        generate_label_columns(df)
        df = normalizing_data(df, scaler)

        #st.dataframe(df)

        ### TimeSeries prep ###
        # Define sequence length
        sequence_length = 50
        # Define column names list
        sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 
                        's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        # Get last sequence (keeping only engines with at least 50 cycles)
        seq_array_last, y_mask = get_last_sequence(df, sequence_length, sequence_cols)
        label_array_last = get_label(df, y_mask)
        # st.write(seq_array_last.shape, label_array_last.shape)

        # Make prediction and store values in a dataframe
        y_true, y_pred = make_class_prediction(seq_array_last, label_array_last)
        pred_df = pd.DataFrame(columns=['engine_id', 'real', 'prediction'])
        pred_df['engine_id'] = engine_remaining
        pred_df['real'] = y_true[:, 0]
        pred_df['prediction'] = y_pred[:, 0]

        ### 15/30 cycles prediction part ###
        st.header("Engine")
        engine_number = st.selectbox('Choose an engine', engine_remaining)
        if engine_number:
            value = pred_df[pred_df['engine_id'] == engine_number]['prediction']

        col1, col2 = st.columns(2)

        with col1:
            st.header("15 cycles")
            if int(value) > 0:
                st.error('Maintenance needed')    
            else:
                st.success('No maintenance needed')

        with col2:
            st.header("30 cycles")
            st.write('TODO')
            #st.success('No maintenance needed') if int(value) > 0 else st.error('Maintenance needed')

        ### Collapsed parts ###
        # Sensors curves
        with st.expander(f"See sensors for engine n° {engine_number}"):
            sensors = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            sensor_visualizer(df, sensors, engine_number)
        
        # Predictions info
        with st.expander("See prediction results"):
            report = classification_report(y_true, y_pred)
            col3, col4 = st.columns([1.5,4])
            with col3:
                st.dataframe(pred_df)
            with col4:
                st.text(report)