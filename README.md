
![yadi](assets/logo/yadi_logo_clear_purple_tilt.png)

# ``yadi``: yet another DSS interface.
A high-level Python interface designed to make it easier to get things done with the [OpenDSS Electric Distribution System Simulator](https://smartgrid.epri.com/SimulationTool.aspx), which is [available for download here](https://sourceforge.net/projects/electricdss/files/) and is designed by the Electric Power Research Institute. We make heavy use of [opendssdirect.py](https://github.com/dss-extensions/OpenDSSDirect.py) and build upon it. 

## Features
``yadi`` gives you a number of tools to improve your distribution system research and accelerate the integration of OpenDSS into your studies. 

### Summary
``yadi`` provides a fresh, modern approach with support for seamless access to several key features crucial for disitrubtion system studies.
#### circuit-based/network-based/model-based analysis
1. Simplified circuit data access, collection, and analysis
2. Time series analysis
3. Sensitivity analysis
5. Electric vehicle analysis
6. Iterative hosting capacity analysis

#### measurement-based/sample-based/model-free analysis
1. Advanced metering infrastructure (AMI) dataset generation
2. Regression-based sensitivity analysis
3. Sensitivity-based hosting capacity generation analysis
4. Sensitivity-based hosting capacity demand analysis

#### benefits:
1. Focus on data-driven/machine learning applications
2. Cross-platform and compatibility with opendssdirect.py
3. Local OpenDSS installation not necessarily required


## Contributors
### Georgia Tech
- Samuel Talkington
- Jorge Fernandez (admittance matrix & initialization in dss/model.py)
- Alex Reyna (dss/ev.py,hc/ev.py,sens/ev.py)

New contributors are always welcome.

## Acknowledgement 
This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2039655. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors(s) and do not necessarily reflect the views of the National Science Foundation.
