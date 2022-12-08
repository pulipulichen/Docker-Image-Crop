FROM tensorflow/tensorflow:2.11.0

RUN apt-get update

RUN pip install keras_ocr
RUN pip install Pillow==9.0.0
RUN pip install autocrop
RUN pip3 install smartcrop
RUN python3 -m pip install --upgrade pip
RUN pip3 install tenserflow

RUN mkdir -p 1.input
RUN mkdir -p 2.inpaint
RUN mkdir -p 3.output

CMD ["python /app/main.py"]