function [Output, new_wieghts] = NeuralNetworkImpl(wieghts, input_layer_size, hidden_layer_size, num_labels, X, y, l)

% ========================= Initialization ============================
Theta1 = reshape(wieghts(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(wieghts((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
m = size(X, 1);
Output = 0;

% ========================= Forward Feed ============================
X = [ones(m, 1), X];
Y = zeros(size(y, 1), num_labels);
for i = 1:num_labels
    Y(:, i) = (y == i);
end

a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2, 1), 1), a2];
a3 = sigmoid(a2 * Theta2');

Y1 = -1 .* (Y .* log(a3));
Y0 = (ones(size(Y)) - Y) .* log(ones(size(a3)) - a3);
R1 = sum(sum(Theta1(:, 2: end) .^ 2));
R2 = sum(sum(Theta2(:, 2: end) .^ 2));
R = R1 + R2;
R = (R * l) / (2 * m);
D = Y1 - Y0;
S = sum(D(:));
Output = (S / m) + R;

% ========================= Back Propagation ============================
d2 = zeros(size(Theta2));
d1 = zeros(size(Theta1));
Theta1_new_wieghts = zeros(size(Theta1));
Theta2_new_wieghts = zeros(size(Theta2));

for i = 1 : m
    a1 = X(i, :);
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2, 1), 1), a2];
    a3 = sigmoid(a2 * Theta2');
    
    s3 = a3 - Y(i, :);
    e = sigmoid(z2);
    g = e .* (ones(size(e)) - e);
    v = s3 * Theta2;
    s2 = v(:, 2:end) .* g;
    
    d2 = d2 + (s3' * a2);
    d1 = d1 + (s2' * a1);
end

% ========================= Wieghts Update ============================
Theta1_new_wieghts = d1 ./ m;
Theta1_new_wieghts(:, 2: end) = Theta1_new_wieghts(:, 2: end)+ ((Theta1(:, 2: end) * l) / m);
Theta2_new_wieghts = d2 ./ m;
Theta2_new_wieghts(:, 2: end) = Theta2_new_wieghts(:, 2: end)+ ((Theta2(:, 2: end) * l) / m);

new_wieghts = [Theta1_new_wieghts(:) ; Theta2_new_wieghts(:)];
end
