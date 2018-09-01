function pred_labels = Prediction_Labels(Theta1, Theta2, X)
m = size(X, 1);

x1 = [ones(m, 1) X];
v1 = x1 * Theta1';
y1 = sigmoid(v1);

x2 = [ones(m, 1) y1];
v2 = x2 * Theta2';
y = sigmoid(v2);

[~, pred_labels] = max(y, [], 2);
end
