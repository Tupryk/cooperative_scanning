import robotic as ry


def get_config(verbose: int=0) -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath("../rai-robotModels/scenarios/dual_floating.g"))
    C.addFile("./banana.g")
    if verbose:
        C.view(True)
    return C
