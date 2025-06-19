% Configure Python environment
pyenv('Version', 'C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe');

% Model configuration
model_file = 'models\model_pytorch.pth'; % Update if different
model_format = 'pth'; % Options: 'pth', 'ckpt', 'onnx'
base_path = 'C:\Users\USER\Downloads\sahan';
model_path = fullfile(base_path, model_file);
data_path = fullfile(base_path, 'X_test.npy');
label_path = fullfile(base_path, 'y_test.npy');

% Verify files exist
if ~exist(data_path, 'file')
    error('X_test.npy not found');
end
if ~exist(label_path, 'file')
    error('y_test.npy not found');
end
if ~exist(model_path, 'file')
    error('%s not found', model_file);
end

% Run prediction with evaluation
[status, cmdout] = system(['python "' base_path '\predict_eeg.py"']);
if status ~= 0
    error('Prediction failed: %s', cmdout);
end

% Run MuJoCo simulation
[status, cmdout] = system(['python "' base_path '\mujoco_control.py"']);
if status ~= 0
    error('MuJoCo simulation failed: %s', cmdout);
end