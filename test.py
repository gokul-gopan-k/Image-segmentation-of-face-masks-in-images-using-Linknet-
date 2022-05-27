from data_prepare import get_data
import matplotlib.pyplot as plt
from train import create_train_model

#Create and train model
model = create_train_model()

#Get test data
X_test_seg,Y_test_seg = get_data(mode = "test")

#Prediction
pred = model.predict(X_test_seg)


#plot prediction and ground truth of test results
fig = plt.figure(figsize=(10, 7))
for i in range(3):
    fig.add_subplot(3, 3, (i*3)+1)
    plt.imshow(X_test_seg[i])
    plt.title("image")
    fig.add_subplot(3, 3, (i*3)+2)
    plt.imshow(Y_test_seg[i])
    plt.title("truth")
    fig.add_subplot(3, 3,(i*3)+3)
    plt.imshow(pred[i])
    plt.title("pred")
