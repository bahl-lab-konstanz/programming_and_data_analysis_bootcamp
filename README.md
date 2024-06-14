# Workshop: programming and data analysis bootcamp

## Objective
The present lecture is intended for you to accomplish the following objectives:
- Learn the basics of _Python_
- Learn how to set up a _conda_ environment
- Get a little experience of scripting using _PyCharm_ IDE
- Run simple data analysis using _pandas_ package
- Have a quick look at a simple model for behavioral tracking data

### Terminology
**python** computer programming language

**interpreter**
program that reads and executes Python code

**PyCharm**
integrated development environment (IDE): software application that helps programmers write, execute and debug code

**module**
a python file: ending with .py

**package**
collection of modules: useful code that has already been written by others

**environment**
directory that contains Python and its packages in a specific combination of versions. Software installed in an environment will only be used inside the environment. It allows to easily isolate Python projects. Gives full control of your project and makes it easily reproducible.

TIP: for every project, create a corresponding environment

**Anaconda**
package manager. It helps you create environments with specific versions of Python and its packages

**pip**
package installer for Python

## Prerequisites
1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) and [PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) (for more step-by-step instructions, use [this guide](https://medium.com/@GalarnykMichael/setting-up-pycharm-with-anaconda-plus-installing-packages-windows-mac-db2b158bd8c))
2. Download this repository as a zip file and unzip it

### Create your python environment
**1. Open Anaconda Prompt**

**2. Create new environment with all the needed dependencies and install python 3.12**

```conda env create --file=<LOCAL_PATH_TO_PROJECT/env_<YOUR_OS>.yaml>```

press `y` and `Enter` if prompted

**3. Activate the new environment to use it**

```conda activate python_bootcamp```

**4. Open PyCharm and add a new interpreter paired with your environment**

**5. (EXTRA) To update a package, e.g. matplotlib, type**

```conda update matplotlib ```

