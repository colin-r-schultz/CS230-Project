import os
import shutil
import pybullet as p
import math

UNIT_LINE = '# File units = '
UNIT_SCALE = {
    'meters': 1,
    'centimeters': 0.01,
    'millimeters': 0.001,
    'inches': 0.0254
}
orient = p.getQuaternionFromEuler([math.pi / 2, 0, 0])

DIR_NAME = 'models'

os.makedirs(DIR_NAME)
subdirs = os.scandir('IKEA')

blacklist = []

blfile = open('blacklist.txt')
for line in blfile:
    blacklist.append(line[:-1])
blfile.close()

dictfile = open(DIR_NAME + '/objlist.txt', 'w')

p.connect(p.DIRECT)

for folder in subdirs:
    if not folder.is_dir():
        continue
    objlist = open(os.path.join(folder.path, 'obj_list.txt'))
    objfile = objlist.readline()[:-1]
    objlist.close()
    if len(objfile) < 1:
        continue
    src = os.path.join(folder.path, objfile + '.obj')
    objname = folder.name
    if objname in blacklist:
        continue
    dst = os.path.join(DIR_NAME, objname + '.obj')
    shutil.copyfile(src, dst)
    dstfile = open(dst)
    units = 'UNKNOWN'
    for line in dstfile:
        if line[0:len(UNIT_LINE)] == UNIT_LINE:
            units = line[len(UNIT_LINE):-1]
            break
    dstfile.close()
    scale = [UNIT_SCALE[units]] * 3
    visShape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=dst, meshScale=scale, 
        visualFrameOrientation=orient)
    colShape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=dst, meshScale=scale,
        collisionFrameOrientation=orient)
    objID = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colShape, baseVisualShapeIndex=visShape)
    aabb = p.getAABB(objID)
    meanX = (aabb[0][0] + aabb[1][0]) / 2
    meanY = (aabb[0][1] + aabb[1][1]) / 2
    minZ = (aabb[0][2])
    dictfile.write('{} {} {},{},{}\n'.format(objname, units, -meanX, -meanY, -minZ))
    print('Copied "{}" to "{}"'.format(src, dst))

dictfile.close()
p.disconnect()
print('Complete!')