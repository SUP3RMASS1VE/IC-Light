# IC-Light Enhanced

This project is an enhanced version of the [IC-Light](https://github.com/lllyasviel/IC-Light) repository, designed for advanced image relighting and enhancement using Stable Diffusion and deep learning techniques.
![Screenshot 2025-03-10 232133](https://github.com/user-attachments/assets/cd2fef8a-bb8c-4e77-8fc8-ca3369cb48f7)
![Screenshot 2025-03-10 232115](https://github.com/user-attachments/assets/723bd314-09ec-4dd9-b268-51dfcae2a7db)

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

