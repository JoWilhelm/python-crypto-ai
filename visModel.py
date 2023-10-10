import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
from tqdm.keras import TqdmCallback
import datetime
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import regularizers
import json
from tensorflow.keras.utils import plot_model
from graphviz import Digraph



inputShape = (180, 3)


batchSize = 96
layers = 2
nodes = 256
denseNodes = 128

dropOut = 0.8 #0.92 #0.88
rec_dropout = 0
l1l2_reg = 0 #1e-5#1e-3

learningRate = 0.00001
decay = 0


# model
model = Sequential()

for _ in range(layers-1):
  model.add(LSTM(nodes, 
               activation="tanh", 
               recurrent_activation = 'sigmoid', 
               recurrent_dropout = rec_dropout, 
               unroll = False, 
               use_bias = True, 
               input_shape=(inputShape), 
               return_sequences=True,
               kernel_regularizer=regularizers.l1_l2(l1=l1l2_reg/10, l2=l1l2_reg),
               #bias_regularizer=regularizers.l2(l1l2_reg),
               activity_regularizer=regularizers.l2(l1l2_reg)
               ))
  model.add(Dropout(dropOut))
  #model.add(BatchNormalization())

model.add(LSTM(nodes, 
             activation="tanh", 
             recurrent_activation = 'sigmoid', 
             recurrent_dropout = rec_dropout, 
             unroll = False, 
             use_bias = True, 
             input_shape=(inputShape),
             kernel_regularizer=regularizers.l1_l2(l1=l1l2_reg/10, l2=l1l2_reg),
             #bias_regularizer=regularizers.l2(l1l2_reg),
             activity_regularizer=regularizers.l2(l1l2_reg)
             ))
model.add(Dropout(dropOut))
#model.add(BatchNormalization())

model.add(Dense(denseNodes, 
                activation="relu",
                kernel_regularizer=regularizers.l1_l2(l1=l1l2_reg/10, l2=l1l2_reg),
                #bias_regularizer=regularizers.l2(l1l2_reg), 
                activity_regularizer=regularizers.l2(l1l2_reg)))
model.add(Dropout(dropOut))

model.add(Dense(3, activation="softmax"))




## Visualisiere das Modell
#plot_model(model, to_file='lstm_rnn.png', show_shapes=True, show_layer_names=True)



# Erstelle ein Diagramm
dot = Digraph(comment='LSTM RNN Architecture', format='png')
dot.attr(rankdir='LR')
dot.node('Input', shape='ellipse', label='Input\n180x3')

dot.node('LSTM1', shape='box', label='LSTM\n180x3 → 180x64')
dot.node('LSTM2', shape='box', label='LSTM\n180x64 → 64')

dot.node('Dropout1', shape='ellipse', label='Dropout', width='0.3', height='0.3', color='dimgrey', fontcolor='dimgrey')
dot.node('Dense1', shape='ellipse', label='Dense\n64 → 32')
dot.node('Dropout2', shape='ellipse', label='Dropout', width='0.3', height='0.3', color='dimgrey', fontcolor='dimgrey')

dot.node('Output', shape='ellipse', label='Output\n32 → 3')

#dot.edges(['Input->LSTM1', 'LSTM1->LSTM2', 'LSTM2->Dropout1', 'Dropout1->Dense1', 'Dense1->Dropout2', 'Dropout2->Output'])
dot.edge('Input', 'LSTM1')
dot.edge('LSTM1', 'LSTM2')
dot.edge('LSTM2', 'Dropout1')
dot.edge('Dropout1', 'Dense1')
dot.edge('Dense1', 'Dropout2')
dot.edge('Dropout2', 'Output')


# Speichere das Diagramm als Bild
dot.render('lstm_rnn', view=True)