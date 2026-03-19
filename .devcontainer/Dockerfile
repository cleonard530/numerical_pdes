FROM fedora:41

RUN dnf install -y python3.12 python3.12-devel git make gcc gcc-c++ diffutils \
    && dnf clean all \
    && python3.12 -m ensurepip --upgrade

COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /workspace
