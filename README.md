# RNN_MusicGen

## TO TRAIN LSTM MODEL AND DUMP TRAINING AND VALIDATION LOSSES ##
python train_lstm.py -l <number_of_hidden_units> -n <number_of_epochs> -m <model_name_to_save_loss> > bg/run_logs.txt 

## TO TRAIN LSTM MODEL AND DUMP TRAINING AND VALIDATION LOSSES ##
python train_rnn.py -l <number_of_hidden_units> -n <number_of_epochs> -m <model_name_to_save_loss> > bg/run_logs.txt 

## TO PLOT TRAINING AND VALIDATION LOSSES ##
plotGraph.ipynb

## TO GENERATE MUSIC AND HEATMAPS ##
sample_lstm.py / sample_lstm.ipynb
