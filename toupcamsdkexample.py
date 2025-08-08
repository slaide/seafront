import toupcam
import numpy as np
import time

class ToupcamController:
    def __init__(self):
        self.camera = None
        self.image_ready = False
        self.latest_image = None
        
    def list_cameras(self):
        """List all available cameras"""
        devices = toupcam.Toupcam.EnumV2()
        print(f"Found {len(devices)} camera(s):")
        for i, device in enumerate(devices):
            print(f"  {i}: {device.displayname} ({device.id})")
        return devices
    
    def open_camera(self, camera_index=0):
        """Open camera by index (0 for first camera)"""
        devices = toupcam.Toupcam.EnumV2()
        if not devices:
            raise Exception("No cameras found")
        
        if camera_index >= len(devices):
            raise Exception(f"Camera index {camera_index} out of range")
            
        # Open camera
        self.camera = toupcam.Toupcam.Open(devices[camera_index].id)
        if not self.camera:
            raise Exception("Failed to open camera")
        print(f"Opened camera: {devices[camera_index].displayname}")
        
        # Get resolution
        width, height = self.camera.get_Size()
        print(f"Resolution: {width}x{height}")
        return width, height
    
    def close_camera(self):
        """Close the camera"""
        if self.camera:
            self.camera.Close()
            self.camera = None
            print("Camera closed")
    
    def set_exposure_and_gain(self, exposure_us=50000, gain=100):
        """Set exposure time (microseconds) and analog gain (percentage)"""
        if not self.camera:
            raise Exception("Camera not opened")
        
        # Set exposure time
        self.camera.put_ExpoTime(exposure_us)
        print(f"Exposure set to {exposure_us} μs")
        
        # Set analog gain
        self.camera.put_ExpoAGain(gain)
        print(f"Gain set to {gain}%")
    
    def event_callback(self, event):
        """Callback for camera events"""
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            self.image_ready = True
    
    def capture_single_image(self):
        """Capture a single image (one-shot mode)"""
        if not self.camera:
            raise Exception("Camera not opened")
        
        try:
            # Start pull mode with callback
            self.camera.StartPullModeWithCallback(self.event_callback, None)
            
            # Wait for image
            timeout = 5.0  # seconds
            start_time = time.time()
            self.image_ready = False
            
            while not self.image_ready and (time.time() - start_time) < timeout:
                time.sleep(0.01)
            
            if not self.image_ready:
                raise Exception("Timeout waiting for image")
            
            # Get image dimensions
            width, height = self.camera.get_Size()
            
            # Pull the image (RGB24 format)
            buf = bytes(width * height * 3)
            self.camera.PullImageV3(buf, 0, 24, 0)
            
            # Convert to numpy array
            image = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            
            # Stop camera
            self.camera.Stop()
            
            print(f"Captured single image: {width}x{height}")
            return image
            
        except Exception as e:
            self.camera.Stop()
            raise e
    
    def start_continuous_acquisition(self, framerate=30):
        """Start continuous image acquisition"""
        if not self.camera:
            raise Exception("Camera not opened")
        
        # Set frame rate limit (0 = no limit)
        if framerate > 0:
            self.camera.put_Option(toupcam.TOUPCAM_OPTION_FRAMERATE, framerate)
            print(f"Frame rate set to {framerate} fps")
        
        # Start pull mode
        self.camera.StartPullModeWithCallback(self.event_callback, None)
        print("Started continuous acquisition")
    
    def get_latest_image(self):
        """Get the latest image from continuous acquisition"""
        if not self.camera:
            raise Exception("Camera not opened")
        
        if not self.image_ready:
            return None
        
        try:
            # Get image dimensions
            width, height = self.camera.get_Size()
            
            # Pull the image (RGB24 format)
            buf = bytes(width * height * 3)
            self.camera.PullImageV3(buf, 0, 24, 0)
            
            # Convert to numpy array
            image = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            
            self.image_ready = False  # Reset flag
            return image
            
        except Exception as e:
            print(f"Error getting image: {e}")
            return None
    
    def stop_continuous_acquisition(self):
        """Stop continuous acquisition"""
        if self.camera:
            self.camera.Stop()
            print("Stopped continuous acquisition")

# Example usage
def main():
    controller = ToupcamController()
    
    try:
        # List cameras
        devices = controller.list_cameras()
        if not devices:
            print("No cameras found")
            return
        
        # Open first camera
        width, height = controller.open_camera(0)
        
        # Set exposure and gain
        controller.set_exposure_and_gain(exposure_us=100000, gain=150)
        
        # Example 1: Single shot capture
        print("\n--- Single Shot Capture ---")
        image = controller.capture_single_image()
        print(f"Image shape: {image.shape}")
        
        # Example 2: Continuous acquisition
        print("\n--- Continuous Acquisition ---")
        controller.start_continuous_acquisition(framerate=10)
        
        # Capture 5 images
        for i in range(5):
            time.sleep(0.2)  # Wait a bit
            img = controller.get_latest_image()
            if img is not None:
                print(f"Got image {i+1}: {img.shape}")
        
        controller.stop_continuous_acquisition()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Always close camera
        controller.close_camera()

if __name__ == "__main__":
    main()
