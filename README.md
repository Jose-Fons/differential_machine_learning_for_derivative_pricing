# differential_machine_learning_for_derivative_pricing

This repository consists on three main scripts (data_generator.py, pricer.py, regulation.py), which are the principal part of my masters' degree final project.
The pricer.py script is where the whole neural newtork is defined, adjusted, trained and setted to predict. This script needs data generated in data_generator.py.
The script regulation.py computes some tests imposed by Basel regulation authorities and shows results.

The other two scripts (KS_graph.py and regulation_with_call) are modifications of the main scripts.
KS_graph.py computes a graphic for the Kolmogorov-Smirnov test with different hyperparameter configuration. This script is a combination of pricer.py and regulation.py that works in a loop to obtain results.
regulation_with_call.py is the same as regulation.py but instead of just using a geometric option as the whole portfolio, it computes a portfolio based on the combination of the geometric option and a call.
