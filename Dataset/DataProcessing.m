images = loadMNISTImages('images.idx3-ubyte');
labels = loadMNISTLabels('labels.idx1-ubyte');
 
disp(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

X = images(1:400,1:5000)';
y = labels(1:5000);
