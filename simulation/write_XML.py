import re
from robotics.task import *
from param_debug import test
from numpy import *
#from sys import exit

def create_xml_lane(xml_lane):
    # shutil can copy the entire folder and rename the copy, I think!
    pass

def write_workspace(ca_tasks, waypoints):

    workspace_xml_text = '<mujoco model="workspace">\n'
    for value, task in enumerate(ca_tasks):
        workspace_xml_text += '    <geom name="obstacle_' + str(value+1)
        workspace_xml_text += '" pos="' + str(asscalar(task.p_0[0])) + ' ' + str(asscalar(task.p_0[1])) + ' ' + str(asscalar(task.p_0[2]))
        workspace_xml_text += '" size="' + str(task.sigma_min) + '" type="sphere" rgba="0.8 0.45 0.15 1" contype="0" conaffinity="0" />\n'
    workspace_xml_text +='\n'
    for value, waypoint in enumerate(waypoints):
        workspace_xml_text += '    <geom name="waypoint_' + str(value+1)
        workspace_xml_text += '" pos="' + str(asscalar(waypoint[0])) + ' ' + str(asscalar(waypoint[1])) + ' ' + str(asscalar(waypoint[2]))
        workspace_xml_text += '" size="0.04" type="sphere" rgba="1 1 1 0.5" contype="0" conaffinity="0" />\n'
    workspace_xml_text += '</mujoco>'

    path = ''
    if not test:
        path = 'simulation/xmls/workspace.xml'
    else:
        path = 'test/xmls/workspace.xml'
    
    with open(path, 'w') as workspace_xml:
        workspace_xml.write(workspace_xml_text)
    

def write_actuators(actuators):

    path = ''
    if not test:
        path = 'simulation/xmls/ur5.xml'
    else:
        path = 'test/xmls/ur5.xml'
    
    data = ""
    with open(path, 'r') as ur5xml_r:
        data = ur5xml_r.read()
    
    matches = re.findall(r'<include file="ur5_.*_actuators\.xml"><\/include>', data)
    if len(matches) <= 0:
        print('Fatal error: actuator xml inclusion not found in ur5.xml using regex, in simulation.write_XML.write_actuators')
        exit(1)
    elif len(matches) > 1:
        print('Fatal error: ambiguous multiple actuator xml inclusions found in ur5.xml using regex, in simulation.write_XML.write_actuators')
        exit(1)

    includestring = matches[0]
        
    if actuators is "position":
        data = data.replace(includestring, '<include file="ur5_position_actuators.xml"></include>')
    elif actuators is "velocity":
        data = data.replace(includestring, '<include file="ur5_velocity_actuators.xml"></include>')
    elif actuators is "motors":
        data = data.replace(includestring, '<include file="ur5_motors_actuators.xml"></include>')
    else:
        print('Error: No actuator type', actuators, 'to write to ur5.xml, in simulation.write_XML.write_actuators')
        exit(1)
        
    with open(path, 'w') as ur5xml_w:
        ur5xml_w.write(data)
