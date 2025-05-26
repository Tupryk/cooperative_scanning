import numpy as np
import robotic as ry


def move_to_look(C: ry.Config, obj_frame_name: str, verbose: int=0) -> np.ndarray:
    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(1, 1, 1, 0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([1.], ry.FS.positionRel, [obj_frame_name, "l_cameraWrist"], ry.OT.eq, [1e1], [0., 0., .3])
    komo.addObjective([1.], ry.FS.positionRel, [obj_frame_name, "r_cameraWrist"], ry.OT.eq, [1e1], [0., 0., .3])

    sol = ry.NLP_Solver()
    sol.setProblem(komo.nlp())
    sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-3, lambdaMax=100., stopInners=20, stopEvals=200)
    ret = sol.solve()
    if not ret.feasible:
        print("KOMO not possible :(")
        exit()

    if verbose:
        komo.view(True)
    
    path = komo.getPath()
    return path
