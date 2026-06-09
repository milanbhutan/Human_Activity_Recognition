The main focus of this project is to repurposes TinyML's magic wand library to perform classification on 

A software flowchart is shown below

<img width="2164" height="1272" alt="image" src="https://github.com/user-attachments/assets/ff054e88-d8f5-41d2-bda4-be6f1965b66a" />

The Arduino code first initializes the IMU by setting the sampling rate and enabling continuous sampling mode. The tensor model is then initialized by including a pointer to its tensor arena and a flatbuffer storing the model parameters. The IMU is then checked to see if data is available, and waits for 20 ms before reading the IMU. Effectively, this makes the IMU sampled at 50 Hz to match how IMU data was collected for the UCI-HAR dataset. For that same reason, we also wait until 128 samples have been collected since our model is trained to classify every 128 IMU samples. These samples are then stored in the model’s input buffer, after which the model will predict the human activity. This is done by choosing the model output that had the largest score.

