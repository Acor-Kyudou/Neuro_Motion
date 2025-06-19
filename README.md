# Neuro_Motion


This projects aims for a simulation using a deep learning model developed using multiple architectures combining together to yield the results and combining the model with MATLAB with integration of MuJoCo for simulation

EEG (Electroencephalogram) is a Brain signal that displays the brain activity in time series data. EEG can be used to help those who lost the connection with brain and their limbs to attain their daily life through Brain-computer interface (BCI). In order to identify the movements based on signal variation. IN order to do the classification CNN is used to identify the non-stationary signals spatial components and then the tranformer acquires global dependancies within output of CNN and MLP header is then used to classify the identified limb movement turning it into a command for Simulation movement. The model accuracy reached the level of 73% with testing while it reached 84% in training and validation.
The modele developed and saved in three different file types saving it as lightning checkpoint, pytorch native and onnx formats

The model pipeline ![image](https://github.com/user-attachments/assets/d9e1ee16-c4c5-41cf-88c2-ff4c62f39f58)


The accuracy and loss of the model is calculated using BCE and dataset was splitted into 81% training, 9% testing and 10% of validation after lot of settings the split in between 30-15-15 splitting to 90-5-5 from the dataset, 81% of data is utilized and imaginary files were used while training after training the model. By using the trained models and real files (where subject actually made the movement) used to test the performance as shown below,

##Training Phase

![Model accuracy while training](https://github.com/user-attachments/assets/81cf7e3a-81bf-4ab3-85bb-d1ae361ca8bc) 

![Classification report](https://github.com/user-attachments/assets/61fdba09-b67a-4d84-b950-2de1b168b7f5)
![COnfusion Matrix when training](https://github.com/user-attachments/assets/4e0889e5-71e6-4e2d-804a-fa1b81a6f927) 

##Now the testing phase
![Model testing accuracy per class](https://github.com/user-attachments/assets/a60acdf6-cd3e-44bf-9568-54170c8823b2) 
![COnfusion Matrix While testing](https://github.com/user-attachments/assets/e68a9c11-c6b9-466e-934a-df1dba6c1af2) 

##References
[Google Deepmind](https://github.com/google-deepmind)
[PhysioNet Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
[Electroencephalogram Signal Classification for action identification](https://keras.io/examples/timeseries/eeg_signal_classification/)
[Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_classification_transformer/)
[Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)
[EEGformer: A transformer–based brain activity classification method using EEG signal](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1148855/full)
