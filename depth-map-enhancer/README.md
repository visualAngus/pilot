# Depth Map Enhancer

## Overview
The Depth Map Enhancer is a Python application designed to generate and enhance depth maps from images. It utilizes advanced depth estimation models to compute depth information and provides users with tools to select specific distance ranges and enhance the contrast of the resulting depth maps.

## Features
- Load images and compute depth maps using state-of-the-art models.
- Select specific distance ranges to focus on particular areas of interest in the depth map.
- Enhance the contrast of the depth maps for better visualization.
- User-friendly interface for easy interaction.

## Project Structure
```
depth-map-enhancer
├── src
│   ├── main.py               # Entry point for the application
│   ├── depth_generator.py     # Contains the DepthMapGenerator class
│   ├── ui
│   │   ├── __init__.py       # UI module initializer
│   │   └── app_window.py      # User interface definitions
│   └── utils
│       ├── __init__.py       # Utils module initializer
│       └── image_processing.py # Utility functions for image processing
├── assets
│   └── sample_images
│       └── .gitkeep          # Keeps the sample_images directory in version control
├── requirements.txt           # Lists project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Specifies files to ignore in version control
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd depth-map-enhancer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/main.py
   ```

2. Use the interface to load an image, select the desired distance range, and enhance the contrast of the depth map.

## Dependencies
- OpenCV
- PyTorch
- torchvision
- numpy
- Any other necessary libraries listed in `requirements.txt`

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.