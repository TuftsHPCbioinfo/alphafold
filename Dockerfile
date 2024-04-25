FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

ENV PATH "/opt/conda/bin:/usr/local/cuda-11.7/bin:/app/:$PATH"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    cuda-command-line-tools-11-7 \
    git \
    hmmer \
    kalign \
    tzdata \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean \
    && mkdir /app \ 
    && cd /app \
    && git clone https://github.com/deepmind/alphafold.git

# Compile HHsuite from source.
RUN /bin/rm -rf /tmp/hh-suite \
    && git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && cd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && cd / \
    && /bin/rm -rf /tmp/hh-suite

# Install Miniconda package manager.
RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
RUN conda install -qy conda==24.1.2 python=3.11 \
    && conda install -y -c conda-forge openmm cudatoolkit==11.7.1 pdbfixer \
    && conda clean --all --force-pkgs-dirs --yes

COPY . /app/alphafold
RUN wget -q -P /app/alphafold/alphafold/common/ \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Install pip packages.
RUN pip3 install --upgrade pip  --no-cache-dir \
    && pip3 install -r /app/alphafold/requirements.txt --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
    jax==0.3.26 \
    jaxlib==0.3.26+cuda11.cudnn805 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Add SETUID bit to the ldconfig binary so that non-root users can run it.


### SETUID bit does not matter: Apptainer does not allow suid commands
RUN chmod u+s /sbin/ldconfig.real
### Workaround below is to use /mnt/out/ld.so.cache for the ld cache file

## Generate run_alphafold.sh

WORKDIR /app/alphafold
RUN echo '#!/bin/bash\n\
ldconfig\n\
python /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
  && chmod +x /app/run_alphafold.sh

ENTRYPOINT ["/app/run_alphafold.sh"]
