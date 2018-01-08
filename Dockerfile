FROM inc0/tf-mkl:latest

RUN apt-get update && apt-get -y install git python-tk
RUN pip install tqdm imageio scikit-image keras ipdb Flask
RUN mkdir -p $(jupyter --data-dir)/nbextensions && \
    cd $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding && \
    jupyter nbextension enable --py widgetsnbextension
RUN mkdir /logs
VOLUME /logs
