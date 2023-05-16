
![yadi](assets/logo/yadi_logo_clear_purple_tilt.png)

# ``yadi``: yet another DSS interface.
`yadi` is software designed to help you tame the black magic that is distribution network models and make decisions with distribution network measurement data. It is primarily built on the [OpenDSS Electric Distribution System Simulator](https://smartgrid.epri.com/SimulationTool.aspx), which is [available for download here](https://sourceforge.net/projects/electricdss/files/) by its authors, the Electric Power Research Institute. We interface with OpenDSS via [opendssdirect.py](https://github.com/dss-extensions/OpenDSSDirect.py) and PyDSS. 


## Quickstart

### Setting up your environment
`yadi` uses [Poetry](https://python-poetry.org/). 

1. Ensure that Poetry is installed. [Follow the instructions for your operating system here.](https://python-poetry.org/docs/basic-usage/)
2. Clone the repository.
3. Navigate to the root directory.
4. While in the root directory, spawn and activate a virtual environment for this project with the command 
`poetry shell`
5. Install the dependencies with 
`poetry install`
6. Confirm that `yadi` is installed in your virtual environment with
`pip show yadi`

### Optional: accessing the virtual envrionment with VS Code
If you use VS Code, after following the above steps, and ensuring that your environment is activated, you can access the virtual environment for ``yadi`` within by running: 
``code .``
within  your poetry shell. You can then select the yadi environment as your python interpreter in the lower right.


## Why `yadi`?

The distribution network modeling community is hiding from modern data science workflows. Many distribution researchers simply implement the same algorithms idiosyncratically over and over again, which has ***severely* limited** the ability of the power system community to develop higher-order iterations of network model analysis tools and algorithms. `yadi` makes it significantly easier to ask the kind of research questions that are useful for the modern, data-driven eletric power system engineer.

`yadi` doesn't stop with making OpenDSS friendlier for modern data science workflows. It also welcomes the [Julia programming language](https://julialang.org/) and its benefits with open arms, and liberally makes uses of `PowerModelsDistribution.jl` in tandem with `OpenDSSDirect.jl` in accordance with each of their strengths. Taking advantage of Julia's speed can allow for sophisticated optimization, decision making, and control algorithms that are not possible in Python alone.

### Features
``yadi`` gives you a number of tools to improve your distribution system research and accelerate the integration of OpenDSS into your studies. ``yadi`` aims to provide a **fresh, modern, open-source approach to the mostly closed-source world of distribution networks** with support for several common algorithms that are central to modern distribution systems research. Below, we summarize some of these features, although it is always a work in progress. Check back in the future for more, or open an issue or pull request. 

#### Turnkey network model analysis tools
1. Simplified circuit data access, collection, and analysis
2. Time series analysis
3. Sensitivity analysis
5. Electric vehicle analysis
6. Iterative hosting capacity analysis

#### Turnkey measurement-based/sample-based/model-free models for the power flow equations
1. Quasi-static time-series (QSTS) dataset generation with user-controllable parameters
2. Regression-based sensitivity analysis
3. Sensitivity-based hosting capacity generation analysis
4. Sensitivity-based hosting capacity demand analysis

## FAQ/Additional benefits:

Not convinced? Here's some more info.

### Main benefits:
1. Focus on data-driven/machine learning applications
2. Cross-platform and compatibility with opendssdirect.py
3. Local OpenDSS installation not necessarily required


## Contributors
### Georgia Tech
- Samuel Talkington 
- Alejandro Owen (conservative linear approximations)
- Alex Reyna
- Jorge Fernandez (admittance matrix & initialization in dss/model.py)

New contributors are always welcome.

## Acknowledgement 
This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2039655. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors(s) and do not necessarily reflect the views of the National Science Foundation.
