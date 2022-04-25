import streamlit as st
import cv2

st.title('First page')

st.markdown(
    """
    ## One more title
    ---
    some text
    """
)

vid = cv2.VideoCapture('/dev/video0')
stframe = st.empty()

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        continue
    
    stframe.image(frame, channels = 'BGR', use_column_width=True)

