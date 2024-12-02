FROM nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt

# maybe we also have a requirements.txt file
# COPY ./requirements.txt /workspace/requirements.txt
# RUN pip install -r requirements.txt

# COPY ./project /workspace/project
# ENTRYPOINT ["python"]
# CMD ["/workspace/project/train.py"] 
ARG WORKSPACE