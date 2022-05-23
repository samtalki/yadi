# yadi: yet another DSS interface.
A high-level Python interface designed to make it easier to get things done with the EPRI OpenDSS Electric Distribution System Simulator. 


## Features
yadi gives you tools to accelerate your integration of OpenDSS into your research and power system studies. It provides support for seamless access to several key features crucial for disitrubtion system studies.

### circuit-based/network-based/model-based analysis
1. Simplified circuit data access, collection, and analysis
2. Time series analysis
3. Voltage sensitivity analysis
5. Electric vehicle analysis
6. Model-based hosting capacity analysis

### measurement-based/sample-based/model-free analysis
1. Advanced metering infrastructure (AMI) dataset generation
2. Regression-based sensitivity analysis
3. Model-free hosting-capacity analysis

### benefits:
1. Cross-platform compatibility
2. Compatibility with opendssdirect.py
3. No OpenDSS installation required


## Contributors

### Georgia Tech
- Samuel Talkington
- Jorge Fernandez (admittance matrix & initialization in dss/model.py)
- Alex Reyna (dss/ev.py,sens/ev_hc.py)