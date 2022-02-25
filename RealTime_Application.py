# TAP libraries
from bleak import _logger as logger
from tapsdk.models import AirGestures
from tapsdk import TapSDK, TapInputMode
import asyncio

# Basic libraries
import math
import numpy as np
import pandas as pd
from matplotlib import *
from sklearn.utils import shuffle

# Dimensionality Reduction libraries
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

# Performance Metrics libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Classifier's libraries
from sklearn import svm

# Webcam libraries
import cv2

# Define global variables
data_array = []
data_dataframe = pd.DataFrame()
classifier = svm.SVC()

############ DATA ACQUISITION ##############
# Upload the dataset from a txt file(the txt file must be edited previously to eliminate the[ and] characters)
df = pd.read_csv('data_tap.txt', delimiter=",")
print(df)
# Shuffle the complete dataset before training
# df = shuffle(df)
# Eliminate the nan values (there are non nan-values, therefore it is not severely handled)
df.dropna()
# Change the string values (the labels) from the column 'symbology' into int values:
df['simbology'] = df['simbology'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J4', 'K', 'L', 'M', 'N',
                                           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '*'],
                                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
true_symbology = df['simbology']

############ PCA Dimensionality Reduction ##############
# Normalize the data
D = df.iloc[:, :-1].to_numpy()
scaler = StandardScaler()
scaler.fit(D)
D_st = scaler.transform(D)
print(D_st)
# Obtain the means of the input data
D_mean = D.mean(axis=0)
print('The meaens are:' + str(D_mean))
# OBtain the covariances od the input data
D_std = D.std(axis=0)
print('The covariances are' + str(D_std))
# Explained variance from the 5th component
pca = decomposition.PCA(n_components=5).fit(D_st)
# Data projected
Xproj = pca.transform(D_st)
print('The shape of the projected data is:' + str(Xproj.shape))
Xproj = pd.DataFrame(Xproj)
print(Xproj)
print(type(Xproj))
Xproj = pd.DataFrame(Xproj)
Xproj['symbols'] = true_symbology
Xproj = shuffle(Xproj)
print('The projected data in dataframe format is:' + '\n' + str(Xproj))
# Keep only the numeric variables to perform the fitting
df_num = Xproj.select_dtypes(include='number')

############ Create the Support Vector Machine classifier ##############
# Define the train dataset (70%) and the test dataset (30%)
[row, col] = df_num.shape
training_rows = math.ceil(row * .7)
testing_rows = math.floor(row * .3)
data_train = df_num.iloc[:training_rows, :col-1]
label_train = df_num.iloc[:training_rows, col-1:]
data_test = df_num.iloc[training_rows:, :col-1]
label_test = df_num.iloc[training_rows:, col-1]
# The following line of code can also be used to do the same data partition:
# trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
# Training the model with the specified classifier
# Choose from the proposed ones [clf_lda, clf_qda, svm, rf...]
classifier.fit(np.array(data_train), np.array(label_train))
# Predict with the test data
label_predicted = classifier.predict(np.array(data_test))
# print(label_predicted)
# Calculate and show the Confusion Matrix
CM_svm = confusion_matrix(label_test, label_predicted)
print('Confusion Matrix:' + '\n' +  str(pd.DataFrame(CM_svm)))
# Calculate and shoe the Accuracy score
accur2 = accuracy_score(label_test, label_predicted)
print('The accuracy of the SVM is: {} %'.format(accur2 * 100))
# Predict just for a simple array of data
print('\n' + 'Predict just for a simple array of data:')
# Without PCA -> test_array = np.array([13, -27, -9, -15, -2, 34, 0, 9, 33, 10, 19, 25, 7, 12, 30])
# With PCA ->
test_array = np.array([56.353360,  36.461163,  -9.610339,  -3.687739,   5.370382])
print('Observation' + str(test_array))
test_array_pred = classifier.predict([test_array])
print('Predicted class:' + str(test_array_pred))

# Define the function that will generate a box containing the predicted letter
def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 5
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

# Libraries, SDK downloaded from the TapStrap device
def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    print("{0}: {1}".format(sender, data))
def OnMouseModeChange(identifier, mouse_mode):
    print(identifier + " changed to mode " + str(mouse_mode))
def OnTapped(identifier, tapcode):
    print(identifier + " tapped " + str(tapcode))
def OnGesture(identifier, gesture):
    print(identifier + " gesture " + str(AirGestures(gesture)))
def OnTapConnected(self, identifier, name, fw):
    print(identifier + " Tap: " + str(name), " FW Version: ", fw)
def OnTapDisconnected(self, identifier):
    print(identifier + " Tap: " + identifier + " disconnected")
def OnMoused(identifier, vx, vy, isMouse):
    print(identifier + " mouse movement: %d, %d, %d" % (vx, vy, isMouse))

# The function that will be used to generate raw data from the hand-movements
# (xyz axis coordinates from every finger with respect to the thumb)
def OnRawData(identifier, packets):
    # Uncomment to acquire data_ Open a txt file where all the data stream from the device will be saved
    # txt = open('data_tap2.txt', 'a')
    # Read only the accelerometer values from all the information send in 1 packet (lots of compressed data)
    for m in packets:
        # If the information received from the packet comes from an accelerometer, then use it
        if m["type"] == "accl":
            # Process the stream of data every 35ms (this value can be modified)
            OnRawData.accl_cnt += 1
            if OnRawData.accl_cnt == 35:
                OnRawData.accl_cnt = 0
                # Save the received data (15 values = xyz * 5 fingers) into an array
                data_array = np.array(m["payload"])
                #The means are:
                means = np.array([21.51181263,-13.28105906,1.1311609,16.97311609,9.96863544,
                                  14.86191446,14.16558045,15.31731161, 9.9790224,12.56639511,
                                  19.10753564,8.14806517,13.82708758,15.22423625,15.64663951])
                # The covariances are
                stds = np.array([10.46414511, 11.02906859,14.72906386, 13.39172048, 11.21374602,
                                 12.21047911, 16.17396756,8.08367176,13.20777744,15.46618317,
                                 5.80464771, 12.52223567,14.47146358,8.32648604,10.06518388])
                # Normalize the new data packet received
                DS = (data_array - means)/stds
                # Apply PCA to the normalized data
                data_proj = pca.transform([DS])
                # To perform the data acquisition, uncomment the next line to write the data into a new line on the .txt
                # txt.write(str(m["payload"]) + ',' + 'R' + '\n')
                # Predict the corresponding number using the previously chosen classifier
                number_pred = classifier.predict(data_proj)
                # Convert the number predicted (array) into a (dataframe) to be able to use the replace function
                number_pred_df= pd.DataFrame(number_pred)
                # Replace the numeric values into strings, which represent the predicted letter from the alphabet
                number_letter = number_pred_df.replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                                                        15,16,17,18,19,20,21,22,23,24,25,26,27],
                                                       ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
                                                        'O','P','Q','R','S','T','U','V','W','X','Y','Z','*'])
                # Print all the desired variables to check if the program works properly
                print('Received data array:')
                print(data_array)
                print('Data projected with PCA:')
                print(data_proj)
                print('Predicted letter:')
                print(str(np.array(number_letter)))
                # Open the webcam and show the predicted letter on the screen to check the classifier's functionality
                ret, frame = cap.read()
                if ret == True:
                    __draw_label(frame, 'Predicted letter:', (20, 390), (255, 255, 255))
                    __draw_label(frame, str(np.array(number_letter)) , (20, 440), (255, 255, 255)) # str(np.array(number_letter))
                cv2.imshow('frame', frame)
                k = cv2.waitKey(5) & 0xff
                # Press the 'esc' key to close the webcam
                if k == 27:
                    break

OnRawData.imu_cnt = 0
OnRawData.accl_cnt = 0
OnRawData.cnt = 0

# Internal function from the Tap device
async def run(loop=None, debug=False):
    if debug:
        import sys

        loop.set_debug(True)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.WARNING)
        logger.addHandler(h)

    client = TapSDK(None, loop)
    if not await client.client.connect_retrieved():
        logger.error("Failed to connect the the Device.")
        return

    logger.info("Connected to {}".format(client.client.address))

    await client.set_input_mode(TapInputMode("controller"))
    await client.register_air_gesture_events(OnGesture)
    await client.register_tap_events(OnTapped)
    await client.register_raw_data_events(OnRawData)
    await client.register_mouse_events(OnMoused)
    await client.register_air_gesture_state_events(OnMouseModeChange)

    # logger.info("Changing to text mode")
    await client.set_input_mode(TapInputMode("text"))
    logger.info("Changing to raw mode")
    await client.set_input_mode(TapInputMode("raw"))
    # await client.send_vibration_sequence([100, 200, 300, 400, 500])
    await asyncio.sleep(50.0, loop=loop)

# Main function from the program
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop, True))
    cap.release()
    cv2.destroyAllWindows()
