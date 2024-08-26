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
> **Note**
> I need to upload the large test data set to Zenodo, but the mammal data set used for testing is included
