import sys
import cv2
import numpy as np
from depth_generator import DepthMapGenerator
from ui.app_window import AppWindow

def main():
    depth_generator = DepthMapGenerator()
    
    # Create the application window
    app = AppWindow(depth_generator)
    
    # Start the application
    app.run()

if __name__ == "__main__":
    main()