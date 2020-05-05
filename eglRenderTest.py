import pybullet as p
import time
import pkgutil
import pybullet_data
import random
import math

def random_color():
    return [random.random(), random.random(), random.random(), 1]

def colorize(obj):
    p.changeVisualShape(obj, -1, rgbaColor=random_color())

p.connect(p.DIRECT)
egl = pkgutil.get_loader('eglRenderer')
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
print("plugin=", plugin)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setGravity(0, 0, -10)
orient = p.getQuaternionFromEuler([0, 0, random.random() * 2 * math.pi])
objID = p.loadURDF("plane.urdf", baseOrientation=orient)
colorize(objID)
p.loadURDF("r2d2.urdf")

pixelWidth = 320
pixelHeight = 220
camTargetPos = [0, 0, 0]
camDistance = 4
pitch = -10.0
roll = 0
upAxisIndex = 2

while (p.isConnected()):
  for yaw in range(0, 360, 10):
    start = time.time()
    p.stepSimulation()
    stop = time.time()
    print("stepSimulation %f" % (stop - start))

    #viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
                                                     upAxisIndex)
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
    ]

    start = time.time()
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix)
    stop = time.time()
    print("renderImage %f" % (stop - start))
    #time.sleep(.1)
    #print("img_arr=",img_arr)

p.unloadPlugin(plugin)
