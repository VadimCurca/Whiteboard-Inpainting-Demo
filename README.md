
# Whiteboard Inpainting Demo

This demo focuses on a whiteboard text overlapped by a person, it detects and hides a person on a video so that all text on a whiteboard is visible.  

## How it works

The input comes from the Raspberry Camera, using Openvino Toolkit it is run through a pre-trained network for person detection.
The network output is the bounding box of the detected person.  
If the person is detected this bounding box is cut and replaced with one on the same location but from a frame with a clean background.  
An edge detection algorithm is also applied to highlight the whiteboard.

### Components

#### Hardware
- Raspberry Pi 4B
- Movidius Nerual Compute Stick 2
- Raspberry Pi Camera

#### Software
- python
- Openvino toolkit
- Streamlit

## How to setup and run

* Some steps to setup [Openvino Toolkit for NCS2](https://docs.openvino.ai/2020.4/openvino_docs_install_guides_installing_openvino_raspbian.html#set-environment-variables)
    * [Some other steps](https://www.hackster.io/news/getting-started-with-the-intel-neural-compute-stick-2-and-the-raspberry-pi-6904ccfe963)
* Install python dependencies from `./requirements_streamlit.txt`
* Go through a few hours of debugging until you realize what else is missing
* Run with `streamlit run ./mySample/project.py`
