import os
import time
import json
import cv2
import numpy as np
import airsim

# --- 配置 ---
OUT_DIR = "quick_check_output"

# --- 主程序 ---
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("Connected to AirSim!")
    except Exception as e:
        raise RuntimeError(f"Could not connect to AirSim. Is it running? Error: {e}")

    vehicles = client.listVehicles()
    if not vehicles:
        raise RuntimeError("No vehicles found.")
    print("Available vehicles:", vehicles)

    for v_name in vehicles:
        print(f"\n--- Capturing data for vehicle: {v_name} ---")
        
        # 1. 获取图像
        try:
            responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)], vehicle_name=v_name)
            response = responses[0]
            
            if response.image_data_uint8:
                img = None
                # 检查是否压缩
                if getattr(response, 'compress', False):
                    img_arr = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                else:
                    img_arr = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    # BGRA 原始数据
                    if img_arr.size == response.width * response.height * 4:
                        img = img_arr.reshape((response.height, response.width, 4))
                        img = img[:, :, :3]  # 丢弃Alpha，转为BGR
                    elif img_arr.size == response.width * response.height * 3:
                        img = img_arr.reshape((response.height, response.width, 3))
                if img is not None:
                    fname = os.path.join(OUT_DIR, f"{v_name}_scene.png")
                    cv2.imwrite(fname, img)
                    print(f"Saved image to {fname}")
                else:
                    print("Failed to decode image.")
            else:
                print("No image data received.")
        except Exception as e:
            print(f"Error getting image: {e}")

        # 2. 获取所有传感器数据
        all_data = {}
        try:
            all_data['state'] = client.getMultirotorState(vehicle_name=v_name)
            all_data['imu'] = client.getImuData(vehicle_name=v_name)
            all_data['barometer'] = client.getBarometerData(vehicle_name=v_name)
            all_data['magnetometer'] = client.getMagnetometerData(vehicle_name=v_name)
            all_data['gps'] = client.getGpsData(vehicle_name=v_name)
            
            # 将AirSim对象转换为字典以便JSON序列化
            printable_data = {key: val.__dict__ for key, val in all_data.items()}
            
            meta_fname = os.path.join(OUT_DIR, f"{v_name}_sensors.json")
            with open(meta_fname, "w", encoding="utf-8") as f:
                json.dump(printable_data, f, ensure_ascii=False, indent=4, default=lambda o: o.__dict__)
            print(f"Saved sensor data to {meta_fname}")

        except Exception as e:
            print(f"Error getting sensor data: {e}")

    print("\nCheck complete. Check the 'quick_check_output' directory.")