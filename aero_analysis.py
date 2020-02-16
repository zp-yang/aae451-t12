from aerosandbox import *
import numpy as np
import matplotlib.pyplot as plt

flat_plat = "naca4412"
main_wing_pos = [0, 0, 0]
mid_wing_pos = [0, 0.2, -0.1]
tip_wing_pos = [0, 0.8, 0.1]

designs = []

l1 = 0.3
l2 = 0.8
theta1 = -0.1
theta2 = np.linspace(0, 0.2)

mid_wing_coords = [0, l1*np.cos(theta1), l1*np.sin(theta1)]
tip_wing_coords = []
for tht in theta2:
    tip_wing_coords.append([0, l2*np.cos(tht), l2*np.sin(tht)])

for i in range(len(tip_wing_coords)):
    designs.append(Airplane(
        name="Peepee hole",
        xyz_ref=[0, 0, 0], # CG location
        wings=[
            Wing(
                name="Main Wing",
                xyz_le=[0, 0, 0], # Coordinates of the wing's leading edge
                symmetric=True,
                xsecs=[ # The wing's cross ("X") sections
                    WingXSec(  # Root
                        xyz_le=[0, 0, 0], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        chord=0.18,
                        twist=0, # degrees
                        airfoil=Airfoil(name=flat_plat),
                        control_surface_type='symmetric',  # Flap # Control surfaces are applied between a given XSec and the next one.
                        control_surface_deflection=0, # degrees
                        control_surface_hinge_point=0.75 # as chord fraction
                    ),
                    WingXSec(  # Mid
                        xyz_le=mid_wing_pos,
                        chord=0.16,
                        twist=0,
                        airfoil=Airfoil(name=flat_plat),
                        control_surface_type='asymmetric',  # Aileron
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(  # Tip
                        xyz_le=tip_wing_coords[i],
                        chord=0.08,
                        twist=0,
                        airfoil=Airfoil(name=flat_plat),
                    )
                ]
            ),
            Wing(
                name="Horizontal Stabilizer",
                xyz_le=[0.6, 0, 0.1],
                symmetric=True,
                xsecs=[
                    WingXSec(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name=flat_plat),
                        control_surface_type='symmetric',  # Elevator
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=0,
                        airfoil=Airfoil(name=flat_plat)
                    )
                ]
            ),
            Wing(
                name="Vertical Stabilizer",
                xyz_le=[0.6, 0, 0.15],
                symmetric=False,
                xsecs=[
                    WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name="naca0003"),
                        control_surface_type='symmetric',  # Rudder
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(
                        xyz_le=[0.04, 0, 0.15],
                        chord=0.06,
                        twist=0,
                        airfoil=Airfoil(name=flat_plat)
                    )
                ]
            )
        ]
    ))

CLs = []
CDs = []

for aircraft in designs:
    aero_problem = vlm3( # Analysis type: Vortex Lattice Method, version 3
        airplane=aircraft,
        op_point=OperatingPoint(
            velocity=5,
            alpha=5,
            beta=0,
            p=0,
            q=0,
            r=0,
        ),
    )
    aero_problem.run() # Runs and prints results to console
    CLs.append(aero_problem.CL)
    CDs.append(aero_problem.CDi)

plt.plot(CLs,CDs)
plt.show()

plt.plot(theta2, CLs)
# aero_problem.draw() # Creates an interactive display of the surface pressures and streamlines

# %%
import casadi as ca

l1 = ca.sym.SX('l1')
l2 = ca.sym.SX('l2')

c = ca.sym.SX('c')

