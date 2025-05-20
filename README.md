# IC-Light Enhanced

This project is an enhanced version of the [IC-Light](https://github.com/lllyasviel/IC-Light) repository, designed for advanced image relighting and enhancement using Stable Diffusion and deep learning techniques.
![Screenshot 2025-05-20 145835](https://github.com/user-attachments/assets/57703773-c81f-4ab1-9061-bfd11aa03718)
![Screenshot 2025-05-20 150246](https://github.com/user-attachments/assets/54093150-f10f-4591-8e82-08c64767613e)

## Features
- Foreground-only relighting
- Foreground and background relighting
- Multiple light source options
- Integration with Stable Diffusion
- Advanced background removal using BriaRMBG

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed, along with the required dependencies:

```sh
pip install -r requirements.txt
```

### Running the Application
To launch the Gradio-based UI, simply run:
```sh
python app.py
```

The app will be available at `http://localhost:7860`.

## Usage
1. Upload an image.
2. Enter a lighting description.
3. Select background options and parameters.
4. Click the "Transform" button to generate the relighted image.

## Acknowledgments
This project is based on the original IC-Light implementation by [lllyasviel](https://github.com/lllyasviel/IC-Light). Special thanks to the original authors for their contributions.

## License
This project follows the original licensing terms of IC-Light. Please refer to their repository for details.

