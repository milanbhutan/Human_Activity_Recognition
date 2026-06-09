/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

#include <cmath>
#include <math.h>

#include "magic_wand_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"


#undef MAGIC_WAND_DEBUG

namespace {

unsigned long last_IMUread = 0;

int8_t QuantizeFloatToInt8(float value, float scale, int zero_point) {
  int32_t q = static_cast<int32_t>(roundf(value / scale)) + zero_point;

  if (q > 127) q = 127;
  if (q < -128) q = -128;

  return static_cast<int8_t>(q);
}

const float CHANNEL_MEAN[6] = {
  8.03577763e-01,
  1.85239076e-02,
  6.81178146e-02,
 -1.96630243e-03,
 -3.20860713e-04,
 -8.10322667e-05
};

const float CHANNEL_STD[6] = {
  0.41372646,
  0.39310044,
  0.36732056,
  0.40898991,
  0.39464557,
  0.26424371
};

const float LPF_ALPHA = 1.0f;

bool filter_initialized = false;

float filt_ax = 0.0f;
float filt_ay = 0.0f;
float filt_az = 0.0f;
float filt_gx = 0.0f;
float filt_gy = 0.0f;
float filt_gz = 0.0f;

void ApplyLowPassFilter(float raw_ax, float raw_ay, float raw_az,
                        float raw_gx, float raw_gy, float raw_gz) {
  if (!filter_initialized) {
    filt_ax = raw_ax;
    filt_ay = raw_ay;
    filt_az = raw_az;
    filt_gx = raw_gx;
    filt_gy = raw_gy;
    filt_gz = raw_gz;
    filter_initialized = true;
    return;
  }

  filt_ax = LPF_ALPHA * raw_ax + (1.0f - LPF_ALPHA) * filt_ax;
  filt_ay = LPF_ALPHA * raw_ay + (1.0f - LPF_ALPHA) * filt_ay;
  filt_az = LPF_ALPHA * raw_az + (1.0f - LPF_ALPHA) * filt_az;

  filt_gx = LPF_ALPHA * raw_gx + (1.0f - LPF_ALPHA) * filt_gx;
  filt_gy = LPF_ALPHA * raw_gy + (1.0f - LPF_ALPHA) * filt_gy;
  filt_gz = LPF_ALPHA * raw_gz + (1.0f - LPF_ALPHA) * filt_gz;
}

// A buffer holding the last 600 sets of 3-channel values from the
// accelerometer.
constexpr int acceleration_data_length = 600 * 3;
float acceleration_data[acceleration_data_length] = {};
// The next free entry in the data array.
int acceleration_data_index = 0;
float acceleration_sample_rate = 0.0f;

// A buffer holding the last 600 sets of 3-channel values from the gyroscope.
constexpr int gyroscope_data_length = 600 * 3;
float gyroscope_data[gyroscope_data_length] = {};
float orientation_data[gyroscope_data_length] = {};
// The next free entry in the data array.
int gyroscope_data_index = 0;
float gyroscope_sample_rate = 0.0f;

float current_velocity[3] = {0.0f, 0.0f, 0.0f};
float current_position[3] = {0.0f, 0.0f, 0.0f};
float current_gravity[3] = {0.0f, 0.0f, 0.0f};
float current_gyroscope_drift[3] = {0.0f, 0.0f, 0.0f};


// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

constexpr int label_count = 6;
const char* labels[label_count] = {"LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"};

void SetupIMU() {
  // Make sure we are pulling measurements into a FIFO.
  // If you see an error on this line, make sure you have at least v1.1.0 of the
  // Arduino_LSM9DS1 library installed.
  IMU.setContinuousMode();

  acceleration_sample_rate = IMU.accelerationSampleRate();
  gyroscope_sample_rate = IMU.gyroscopeSampleRate();

}

void ReadAccelerometerAndGyroscope(int* new_accelerometer_samples,
                                   int* new_gyroscope_samples) {
  *new_accelerometer_samples = 0;
  *new_gyroscope_samples = 0;

    float raw_ax;
    float raw_ay;
    float raw_az;

    float raw_gx;
    float raw_gy;
    float raw_gz;


    if (!IMU.readAcceleration(raw_ax, raw_ay, raw_az)) {
      MicroPrintf("Failed to read acceleration data");
      return;
    }

    if (!IMU.readGyroscope(raw_gx, raw_gy, raw_gz)) {
      MicroPrintf("Failed to read gyroscope data");
      return;
    }

    raw_gx*=DEG_TO_RAD;
    raw_gy*=DEG_TO_RAD;
    raw_gz*=DEG_TO_RAD;

    ApplyLowPassFilter(raw_ax, raw_ay, raw_az,
                       raw_gx, raw_gy, raw_gz);

    const int acceleration_index =
        acceleration_data_index % acceleration_data_length;

    acceleration_data[acceleration_index + 0] = filt_ax;
    acceleration_data[acceleration_index + 1] = filt_ay;
    acceleration_data[acceleration_index + 2] = filt_az;

    acceleration_data_index += 3;
    *new_accelerometer_samples += 1;

    const int gyroscope_index =
        gyroscope_data_index % gyroscope_data_length;

    gyroscope_data[gyroscope_index + 0] = filt_gx;
    gyroscope_data[gyroscope_index + 1] = filt_gy;
    gyroscope_data[gyroscope_index + 2] = filt_gz;

    gyroscope_data_index += 3;
    *new_gyroscope_samples += 1;
  
}

float VectorMagnitude(const float* vec) {
  const float x = vec[0];
  const float y = vec[1];
  const float z = vec[2];
  return sqrtf((x * x) + (y * y) + (z * z));
}

void EstimateGravityDirection(float* gravity) {
  int samples_to_average = 100;
  if (samples_to_average >= acceleration_data_index) {
    samples_to_average = acceleration_data_index;
  }

  const int start_index =
      ((acceleration_data_index +
        (acceleration_data_length - (3 * (samples_to_average + 1)))) %
       acceleration_data_length);

  float x_total = 0.0f;
  float y_total = 0.0f;
  float z_total = 0.0f;
  for (int i = 0; i < samples_to_average; ++i) {
    const int index = ((start_index + (i * 3)) % acceleration_data_length);
    const float* entry = &acceleration_data[index];
    const float x = entry[0];
    const float y = entry[1];
    const float z = entry[2];
    x_total += x;
    y_total += y;
    z_total += z;
  }
  gravity[0] = x_total / samples_to_average;
  gravity[1] = y_total / samples_to_average;
  gravity[2] = z_total / samples_to_average;
}

void UpdateVelocity(int new_samples, float* gravity) {
  const float gravity_x = gravity[0];
  const float gravity_y = gravity[1];
  const float gravity_z = gravity[2];

  const int start_index =
      ((acceleration_data_index +
        (acceleration_data_length - (3 * (new_samples + 1)))) %
       acceleration_data_length);

  const float friction_fudge = 0.98f;

  for (int i = 0; i < new_samples; ++i) {
    const int index = ((start_index + (i * 3)) % acceleration_data_length);
    const float* entry = &acceleration_data[index];
    const float ax = entry[0];
    const float ay = entry[1];
    const float az = entry[2];

    // Try to remove gravity from the raw acceleration values.
    const float ax_minus_gravity = ax - gravity_x;
    const float ay_minus_gravity = ay - gravity_y;
    const float az_minus_gravity = az - gravity_z;

    // Update velocity based on the normalized acceleration.
    current_velocity[0] += ax_minus_gravity;
    current_velocity[1] += ay_minus_gravity;
    current_velocity[2] += az_minus_gravity;

    // Dampen the velocity slightly with a fudge factor to stop it exploding.
    current_velocity[0] *= friction_fudge;
    current_velocity[1] *= friction_fudge;
    current_velocity[2] *= friction_fudge;

    // Update the position estimate based on the velocity.
    current_position[0] += current_velocity[0];
    current_position[1] += current_velocity[1];
    current_position[2] += current_velocity[2];
  }
}

void EstimateGyroscopeDrift(float* drift) {
  const bool isMoving = VectorMagnitude(current_velocity) > 0.1f;
  if (isMoving) {
    return;
  }

  int samples_to_average = 20;
  if (samples_to_average >= gyroscope_data_index) {
    samples_to_average = gyroscope_data_index;
  }

  const int start_index =
      ((gyroscope_data_index +
        (gyroscope_data_length - (3 * (samples_to_average + 1)))) %
       gyroscope_data_length);

  float x_total = 0.0f;
  float y_total = 0.0f;
  float z_total = 0.0f;
  for (int i = 0; i < samples_to_average; ++i) {
    const int index = ((start_index + (i * 3)) % gyroscope_data_length);
    const float* entry = &gyroscope_data[index];
    const float x = entry[0];
    const float y = entry[1];
    const float z = entry[2];
    x_total += x;
    y_total += y;
    z_total += z;
  }
  drift[0] = x_total / samples_to_average;
  drift[1] = y_total / samples_to_average;
  drift[2] = z_total / samples_to_average;
}

void UpdateOrientation(int new_samples, float* gravity, float* drift) {
  const float drift_x = drift[0];
  const float drift_y = drift[1];
  const float drift_z = drift[2];

  const int start_index =
      ((gyroscope_data_index + (gyroscope_data_length - (3 * new_samples))) %
       gyroscope_data_length);

  // The gyroscope values are in degrees-per-second, so to approximate
  // degrees in the integrated orientation, we need to divide each value
  // by the number of samples each second.
  const float recip_sample_rate = 1.0f / gyroscope_sample_rate;

  for (int i = 0; i < new_samples; ++i) {
    const int index = ((start_index + (i * 3)) % gyroscope_data_length);
    const float* entry = &gyroscope_data[index];
    const float dx = entry[0];
    const float dy = entry[1];
    const float dz = entry[2];

    // Try to remove sensor errors from the raw gyroscope values.
    const float dx_minus_drift = dx - drift_x;
    const float dy_minus_drift = dy - drift_y;
    const float dz_minus_drift = dz - drift_z;

    // Convert from degrees-per-second to appropriate units for this
    // time interval.
    const float dx_normalized = dx_minus_drift * recip_sample_rate;
    const float dy_normalized = dy_minus_drift * recip_sample_rate;
    const float dz_normalized = dz_minus_drift * recip_sample_rate;

    // Update orientation based on the gyroscope data.
    float* current_orientation = &orientation_data[index];
    const int previous_index =
        (index + (gyroscope_data_length - 3)) % gyroscope_data_length;
    const float* previous_orientation = &orientation_data[previous_index];
    current_orientation[0] = previous_orientation[0] + dx_normalized;
    current_orientation[1] = previous_orientation[1] + dy_normalized;
    current_orientation[2] = previous_orientation[2] + dz_normalized;
  }
}

      
    
  


}  // namespace

void setup() {
  tflite::InitializeTarget();  // setup serial port

  MicroPrintf("Started");

  if (!IMU.begin()) {
    MicroPrintf("Failed to initialized IMU!");
    while (true) {
      // NORETURN
    }
  }

  SetupIMU();


 

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  
  // static tflite::MicroMutableOpResolver<6> micro_op_resolver;  // NOLINT
  // micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddMean();
  // micro_op_resolver.AddFullyConnected();
  // micro_op_resolver.AddSoftmax();
  // micro_op_resolver.AddExpandDims();
  // micro_op_resolver.AddReshape();
  // micro_op_resolver.AddDepthwiseConv2D();

  static tflite::AllOpsResolver micro_op_resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  TfLiteTensor* model_input = interpreter->input(0);
  TfLiteTensor* model_output = interpreter->output(0);

  Serial.println("Input size");
  for(int i=0;i<model_input->dims->size;i++){
    Serial.println(model_input->dims->data[i]);

  }
  Serial.println("Output size");
  for(int i=0;i<model_output->dims->size;i++){
    Serial.println(model_output->dims->data[i]);

  }
if ((model_input->dims->size != 3) ||
    (model_input->dims->data[0] != 1) ||
    (model_input->dims->data[1] != 128) ||
    (model_input->dims->data[2] != 6) ||
    (model_input->type != kTfLiteInt8)) {
  MicroPrintf("Bad input tensor parameters in model");
  return;
}

 
  if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != label_count) ||
      (model_output->type != kTfLiteInt8)) {
    //MicroPrintf("Bad output tensor parameters in model");
    return;
  }
}

void loop() {


  const bool data_available =
      IMU.accelerationAvailable() && IMU.gyroscopeAvailable();
  if (!data_available) {
    return;
  }

  int accelerometer_samples_read;
  int gyroscope_samples_read;
  
  if ((millis()-last_IMUread)<20){
    return;
  }
  
  last_IMUread=millis();
  
  ReadAccelerometerAndGyroscope(&accelerometer_samples_read,
                                &gyroscope_samples_read);



  bool done_just_triggered = false;
  if (gyroscope_samples_read > 0) {
    EstimateGyroscopeDrift(current_gyroscope_drift);
    UpdateOrientation(gyroscope_samples_read, current_gravity,
                      current_gyroscope_drift);
  }

  if (accelerometer_samples_read > 0) {
    EstimateGravityDirection(current_gravity);
    UpdateVelocity(accelerometer_samples_read, current_gravity);
  }
    

    if ((acceleration_data_index < 128 * 3) ||
    (gyroscope_data_index < 128 * 3)) {
  MicroPrintf("Waiting for 128 IMU samples...");
  return;
}

Serial.print("latest raw: ");

    Serial.print(acceleration_data[(acceleration_data_index - 3 + acceleration_data_length) % acceleration_data_length], 4);
Serial.print(" ");
Serial.print(acceleration_data[(acceleration_data_index - 2 + acceleration_data_length) % acceleration_data_length], 4);
Serial.print(" ");
Serial.print(acceleration_data[(acceleration_data_index - 1 + acceleration_data_length) % acceleration_data_length], 4);
Serial.print(" ");
Serial.print(gyroscope_data[(gyroscope_data_index - 3 + gyroscope_data_length) % gyroscope_data_length], 4);
Serial.print(" ");
Serial.print(gyroscope_data[(gyroscope_data_index - 2 + gyroscope_data_length) % gyroscope_data_length], 4);
Serial.print(" ");
Serial.println(gyroscope_data[(gyroscope_data_index - 1 + gyroscope_data_length) % gyroscope_data_length], 4);




TfLiteTensor* model_input = interpreter->input(0);
float scale = model_input->params.scale;
int zero_point = model_input->params.zero_point;

for (int t = 0; t < 128; t++) {
  int accel_index = (acceleration_data_index - (128 * 3) + (t * 3) + acceleration_data_length) %
                    acceleration_data_length;

  int gyro_index = (gyroscope_data_index - (128 * 3) + (t * 3) + gyroscope_data_length) %
                   gyroscope_data_length;

float raw_ax = acceleration_data[accel_index + 0];
float raw_ay = acceleration_data[accel_index + 1];
float raw_az = acceleration_data[accel_index + 2];



float raw_gx = gyroscope_data[gyro_index + 0];
float raw_gy = gyroscope_data[gyro_index + 1];
float raw_gz = gyroscope_data[gyro_index + 2];


// Normalize using the same mean/std from Python training
float ax = (raw_ax - CHANNEL_MEAN[0]) / CHANNEL_STD[0];
float ay = (raw_ay - CHANNEL_MEAN[1]) / CHANNEL_STD[1];
float az = (raw_az - CHANNEL_MEAN[2]) / CHANNEL_STD[2];

float gx = (raw_gx - CHANNEL_MEAN[3]) / CHANNEL_STD[3];
float gy = (raw_gy - CHANNEL_MEAN[4]) / CHANNEL_STD[4];
float gz = (raw_gz - CHANNEL_MEAN[5]) / CHANNEL_STD[5];

  model_input->data.int8[t * 6 + 0] = QuantizeFloatToInt8(ax, scale, zero_point);
  model_input->data.int8[t * 6 + 1] = QuantizeFloatToInt8(ay, scale, zero_point);
  model_input->data.int8[t * 6 + 2] = QuantizeFloatToInt8(az, scale, zero_point);
  model_input->data.int8[t * 6 + 3] = QuantizeFloatToInt8(gx, scale, zero_point);
  model_input->data.int8[t * 6 + 4] = QuantizeFloatToInt8(gy, scale, zero_point);
  model_input->data.int8[t * 6 + 5] = QuantizeFloatToInt8(gz, scale, zero_point);
}

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed");
      return;
    }


    int8_t max_score;
    int max_index;
TfLiteTensor* output = interpreter->output(0);

Serial.print("Output scale = ");
Serial.println(output->params.scale, 8);

Serial.print("Output zero point = ");
Serial.println(output->params.zero_point);

int best_index = 0;
int8_t best_raw = output->data.int8[0];

for (int i = 0; i < label_count; i++) {
  int8_t raw = output->data.int8[i];
  float deq = (raw - output->params.zero_point) * output->params.scale;

  Serial.print(labels[i]);
  Serial.print(" raw=");
  Serial.print((int)raw);
  Serial.print(" deq=");
  Serial.println(deq, 6);

  if (raw > best_raw) {
    best_raw = raw;
    best_index = i;
  }
}

Serial.print("PREDICTED: ");
Serial.println(labels[best_index]);
Serial.println();
}
