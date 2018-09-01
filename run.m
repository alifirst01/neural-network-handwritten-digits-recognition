%% Initialization
clear ; close all; clc

input_layer_size  = 400;  % 20x20 pixels of train_data images as features of images
hidden_layer_size = 25;   % No. of Neurons in hidden layer
num_labels = 10;          % Labels from 0 - 9
iterations = 100;
rate = 0.3;

load('data.mat');
idx = randperm(size(X, 1));
train_data = X(idx(1:4000),:);
train_data_labels = y(idx(1:4000));
test_data = X(idx(4001:5000),:);
test_data_labels = y(idx(4001:5000));
m = size(train_data, 1);

% ================== Initializing Weights Phase ====================
initial_Theta1 = InitializeWeights(input_layer_size, hidden_layer_size);  % Randomltrain_data_labels Initializing Weights
initial_Theta2 = InitializeWeights(hidden_layer_size, num_labels);
initial_parameters = [initial_Theta1(:) ; initial_Theta2(:)];                  % Concatinating both latrain_data_labelsers' weights to pass in function


% ======================== Training Phase ===========================
options = optimset('MaxIter', iterations);
NN_Function = @(p) NeuralNetworkImpl(p, input_layer_size, hidden_layer_size, ...
                                     num_labels, train_data, train_data_labels, rate);

[parameters, cost] = fmincg(NN_Function, initial_parameters, options);

Theta1 = reshape(parameters(1:hidden_layer_size * (input_layer_size+ 1)), hidden_layer_size, (input_layer_size+ 1));
Theta2 = reshape(parameters((1 + (hidden_layer_size * (input_layer_size+ 1))):end), num_labels, (hidden_layer_size+ 1));
             
   
% ======================== Testing Phase ===========================
pred = Prediction_Labels(Theta1, Theta2, train_data);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == train_data_labels)) * 100);

pred = Prediction_Labels(Theta1, Theta2, test_data);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == test_data_labels)) * 100);

