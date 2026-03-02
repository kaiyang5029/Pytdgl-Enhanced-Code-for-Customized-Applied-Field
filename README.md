# Pytdgl-Enhanced-Code-for-Customized-Applied-Field
This code extended the old pytdgl code from uniform applied field to customized field with demonstration.

Main Scripts:

1. Main_computation.ipynb:
    To perform simulation.
    Follow pytdgl website instructions for the implementation.

2. B_demag_visualization.ipynb: 
    To visualize the applied field.
    Prepare a 2d .np field file to visualize.


Supporting Scripts:
    Here contains updated functions by Justin and kaiyang.
    We took the functions from pytdgl github and changed some code.

1. Animation.py
    Include domain wall into visualization.

2. fmfield_z.py
    Turn our mumax3 simulated field into pytdgl compatible field.
    Prepare a 2d .np field file.

3. Time_dependent_field.py
    A simple demonstration on implementing time-dependent uniform field.
    It should be combined with fmfield_z.py to generate more complicated time-dependent field distribution.


Steps:

1. Generate field on mumax3 with the intended shape.
2. Use helper_code by Justin or Xiaoye to turn mumax3 ovf file into np file. (m, image, outline)
3. Send np file into B_demag_visualization.py to confirm.
4. Setup the correct file name in fmfield_z.py and the outline file name in Main_computation.ipynb.
5. Generate SC and incorporate fmfield_z field in Main_computation.ipynb to run simulation.
6. Display vorticity and order parameter.
