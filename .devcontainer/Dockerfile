FROM python:3.8.13

RUN apt-get update && apt-get install -y nano tig

COPY /requirements /requirements
COPY /requirements-dev /requirements-dev
RUN pip install -r /requirements-dev/requirements-dev.txt
RUN pip install -r /requirements/requirements.txt

RUN curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash
RUN inv --print-completion-script bash > ~/.invoke-completion.sh
COPY .devcontainer/bashrc_content /root/.bashrc