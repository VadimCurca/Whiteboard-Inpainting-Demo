from __future__ import print_function
import streamlit as st
import cv2
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser


st.title('Person detection')

st.markdown(
    """
    ---
    """
)

stframe = st.empty()
stframe2 = st.empty()

# --------------------------------------------------

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Settings')

st.sidebar.markdown('---')

personInpaint = st.sidebar.checkbox("Enable person inpaint", value=True)
drawBox = st.sidebar.checkbox("Enable box drawing around prediction", value=False)
showOrigFrame = st.sidebar.checkbox("Show original frame", value=False)

st.sidebar.markdown('---')

edgeDetection = st.sidebar.checkbox("Enable edge detection", value=False)

cannyMaxValue = st.sidebar.slider('Edge detection: Canny max value', min_value = 0, max_value = 400, value = 200)
cannyMinValue = st.sidebar.slider('Edge detection: Canny min value', min_value = 0, max_value = 400, value = 100)

st.sidebar.markdown('---')

predictionThreshold = st.sidebar.slider('Prediction threshold', min_value = 0.0, max_value = 1.0, value = 0.5)
maxPredicted = st.sidebar.number_input('Maximum number of persons', value=2, min_value=1)

# --------------------------------------------------

#@st.cache()
@st.cache(suppress_st_warning=True)
def run_infer():
    model_xml = "../l_openvino_toolkit_runtime_raspbian_p_2020.4.287/models/person-detection-retail-0013/FP16/person-detection-retail-0013.xml"
    model_bin = "../l_openvino_toolkit_runtime_raspbian_p_2020.4.287/models/person-detection-retail-0013/FP16/person-detection-retail-0013.bin"
    device = "MYRIAD"
    prob_threshold = predictionThreshold

    plugin = IEPlugin(device=device, plugin_dirs=None)

    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    exec_net = plugin.load(network=net, num_requests=2)

    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net

    labels_map = None

    cap = cv2.VideoCapture('/dev/video0')

    ret, frame = cap.read()

    clean_frame = frame.copy()

    cur_request_id = 0
    next_request_id = 1

    while cap.isOpened():
        ret, next_frame = cap.read()

        if not ret:
            continue

        initial_w = cap.get(3)
        initial_h = cap.get(4)

        inf_start = time.time()

        in_frame = cv2.resize(next_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1)) # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})

        if(exec_net.requests[cur_request_id].wait(-1) == 0):
            inf_end = time.time()
            det_time = inf_end - inf_start

        # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            predicted = 0
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)

                    xmin -= 50
                    ymin -= 50
                    xmax += 50
                    ymax += 50

                    if personInpaint:
                        frame_crop = frame[ymin:ymax, xmin:xmax]
                        clean_frame_crop = clean_frame[ymin:ymax, xmin:xmax]
                        frame_crop[:] = clean_frame_crop
                        # clean_frame = frame.copy()
                        clean_frame = frame

                    if drawBox:
                        class_id = int(obj[1])
                        # Draw box and label\class_id
                        color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        det_label = labels_map[class_id] if labels_map else str(class_id)
                        cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

                    predicted += 1
                    if predicted >= maxPredicted:
                        break


        #

        channels = 'BGR'

        if edgeDetection:
            frame = cv2.Canny(frame, cannyMinValue, cannyMaxValue)
            channels = 'BGRA'
        stframe.image(frame, channels = channels, use_column_width=True)

        if showOrigFrame:
            stframe2.image(next_frame, channels = 'BGR', use_column_width=True)

        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

# --------------------------------------------------

run_infer()
