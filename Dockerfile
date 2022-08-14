# Dockerfile, Image, Container

FROM continuumio/miniconda3

LABEL maintainer="Gaurav Rana"

ADD HousePricePrediction .

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create environment
RUN conda env update -f deploy/conda/env.yml

# Make RUN command to use new environment
# SHELL ["conda", "run", "-n", "mle-dev", "bin/bash", "-c"]

CMD [ "bash", "run.sh" ]
