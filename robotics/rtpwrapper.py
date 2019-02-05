from numpy import *

# Robotics Toolbox Python
from .rtp.Link import *
from .rtp.Robot import *
from .rtp.jacobian import *
from .ur5_param import d1, d4, d5, d6, a2, a3, alpha1, alpha4, alpha5

class rtpUR5:
    
    def __init__(self):

        UR5L = []
        UR5L.append( Link(alpha=alpha1,  A=0,      D=d1    ) )
        UR5L.append( Link(alpha=0,       A=a2,     D=0     ) )
        UR5L.append( Link(alpha=0,       A=a3,     D=0     ) )
        UR5L.append( Link(alpha=alpha4,  A=0,      D=d4 ) )
        UR5L.append( Link(alpha=alpha5,  A=0,      D=d5 ) )
        UR5L.append( Link(alpha=0,       A=0,      D=d6 ) )
    
        self.UR5 = Robot(UR5L)
    
    def Jpos(self, configuration):
        return array(jacob0(self.UR5, configuration))
    def fpos(self, configuration):
        pos = array(fkine(self.UR5, configuration)[0:3, 3])
        return pos
    #def rt(self, link_num, configuration):
    #    return self.UR5.links[link_num].tr(configuration[link_num])
    def iconf(self, target, guess_configuration):
        tr = eye(4)
        tr[:3, 3] = target[:, 0]
        print(tr)
        estimated_configuration = ikine(self.UR5, tr, q0=guess_configuration)
        return estimated_configuration
