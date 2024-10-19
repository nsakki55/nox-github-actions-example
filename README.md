# Run Nox using GitHub Actions
## About Nox
Nox is a flexible test automation tool for python.  
https://nox.thea.codes/en/stable/ 

In this repository, you can see the example for how to run nox  in github actions.

## Install Nox through uv
In this project, python package is managed by uv.  install uv when it's not installed yet.
```sh 
$ pip install uv
```

install nox as uv tool
```sh
$ uv tool install nox
```

## Run Nox
Run all nox session
```sh
$ uvx nox
```

Run nox session with test tag
```sh
$ uvx nox -t test
```

Run nox session with lint tag
```sh
$ uvx nox -t lint
```

Run nox session with passing arguments
```sh
$ uvx nox -t test -- tests/test_train_lr.py
```

## Run Nox in GitHub Actions
[nox_test.yml](.github/workflows/nox_test.yml) is demonstrating how nox works in github actions.