# mle-training

Code for Assignments

# Docker installation

1. Docker used in root
    `sudo -i`

2. In WSL to start docker,
    `sudo service docker start`

3. `docker run -dit debian`
    - 'd' denotes **detach**, means it can run in background
    - 'i' denotes **interactive**
    - 't' denotes allocates sudo terminal

4. Build docker image using,

    `DOCKER_BUILDKIT=1 docker build -t <tigerID>/<project_name>:<tag> <source>`

5. Run docker image in bash,

    `docker run --network='host' -it <doker details> bin/bash`

