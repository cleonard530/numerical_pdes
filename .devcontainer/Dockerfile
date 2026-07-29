FROM fedora:42

RUN dnf install -y python3.12 python3.12-devel git make gcc gcc-c++ diffutils cmake gdb file \
    && dnf clean all \
    && python3.12 -m ensurepip --upgrade

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --python python3.12 --no-cache -r /tmp/requirements.txt \
    && uv pip install --system --python python3.12 --no-cache torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && TORCH_CMAKE_PREFIX="$(python3.12 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    && ln -sfn "${TORCH_CMAKE_PREFIX}" /usr/local/share/torch-cmake \
    && rm /tmp/requirements.txt

# So find_package(Torch) works without -DCMAKE_PREFIX_PATH=...
ENV CMAKE_PREFIX_PATH=/usr/local/share/torch-cmake

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY .devcontainer/ /tmp/devcontainer/
RUN if [ -f /tmp/devcontainer/bashrc-extra.sh ]; then \
        cat /tmp/devcontainer/bashrc-extra.sh >> /root/.bashrc; \
    fi \
    && rm -rf /tmp/devcontainer

WORKDIR /workspace
