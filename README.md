## Project Overview

The main focus of this project is to repurposes TinyML's magic wand library to perform classification on d

## Software Architecture 

<img width="2164" height="1272" alt="image" src="https://github.com/user-attachments/assets/ff054e88-d8f5-41d2-bda4-be6f1965b66a" />

The Arduino code first initializes the IMU by setting the sampling rate and enabling continuous sampling mode. The tensor model is then initialized by including a pointer to its tensor arena and a flatbuffer storing the model parameters. The IMU is then checked to see if data is available, and waits for 20 ms before reading the IMU. Effectively, this makes the IMU sampled at 50 Hz to match how IMU data was collected for the UCI-HAR dataset. For that same reason, we also wait until 128 samples have been collected since our model is trained to classify every 128 IMU samples. These samples are then stored in the model’s input buffer, after which the model will predict the human activity. This is done by choosing the model output that had the largest score.

## Getting Started

### Prerequisites
- [VS Code](https://code.visualstudio.com/) with the [PlatformIO extension](https://platformio.org/install/ide?install=vscode) installed
- Arduino Nano 33 BLE Sense
- USB cable to connect the board

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/milanbhutan/Human_Activity_Recognition.git
```

2. **Open in PlatformIO**
   - Open VS Code
   - Click **File → Open Folder** and select the cloned repo folder
   - PlatformIO will automatically detect the project

3. **Install dependencies**
   PlatformIO will automatically download the TensorFlow Lite Micro library on first build — no manual steps needed.

4. **Connect your board**
   Plug in your Arduino Nano 33 BLE Sense via USB.

5. **Build and upload**
   Click the **Upload** button (→) in the PlatformIO toolbar, or run:
```bash
   pio run --target upload
```

6. **View predictions**
   Open the Serial Monitor at **9600 baud** to see live activity predictions.

### Note on Board Compatibility
This project is designed specifically for the **Arduino Nano 33 BLE Sense**, as the IMU reading code targets the onboard LSM9DS1 sensor. Porting to another board would require rewriting the IMU interface code, not just changing the board in `platformio.ini`.
