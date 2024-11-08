# Building the manuscript

In the root of the repo

```
$ docker run --rm \
    --volume $PWD/paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara \
    && open paper/paper.pdf
```
