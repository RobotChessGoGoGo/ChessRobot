import numpy as np
from math import atan2, sqrt, cos, sin, acos, pi, copysign
import time
import os
from gtts import gTTS
import ChessRobot.VisionModule as vm
import ChessRobot.lss as lss
import ChessRobot.lss_const as lssc
import platform
import ChessRobot.Interface as inter
import pdb

if platform.system() == 'Windows':
    port = "COM5"
elif platform.system() == 'Linux':
    port = "/dev/ttyUSB0"
    
lss.initBus(port,lssc.LSS_DefaultBaud) 
base = lss.LSS(1)
shoulder = lss.LSS(2)
elbow = lss.LSS(3)
wrist = lss.LSS(4)
gripper = lss.LSS(5)
allMotors = lss.LSS(254)

allMotors.setAngularStiffness(0)
allMotors.setAngularHoldingStiffness(0)
allMotors.setMaxSpeed(30)

gripper.setMaxSpeed(1200)

CST_ANGLE_MIN = -90
CST_ANGLE_MAX = 90

def checkConstraints(value, min, max):
    if (value < min):
        value = min
    if (value > max):
        value = max
    return (value)

# Desired positions in x, y, z, gripper aperture
def LSS_IK(targetXYZG, gripOffset):

    d1 = 4.055   # Bottom to shoulder
    d2 = 5.571   # Shoulder to elbow
    d3 = 6.425   # Elbow to wrist
    d4 = 4.52   # Wrist to end of gripper

    x0 = targetXYZG[0]
    y0 = targetXYZG[1]
    z0 = targetXYZG[2]
    g0 = targetXYZG[3]

    # Base angle (degrees)
    q1 = atan2(y0,x0) * 180/pi

    # Radius from the axis of rotation of the base in xy plane
    xyr = sqrt(x0**2 + y0**2)

    # Pitch angle    
    q0 = 80

    # Gripper components in xz plane
    lx = d4 * cos(q0 * pi/180)
    lz = d4 * sin(q0 * pi/180)

    # Wrist coordinates in xz plane
    x1 = xyr - lx
    z1 = z0 + lz - d1

    # Distance between the shoulder axis and the wrist axis
    h = sqrt(x1**2 + z1**2)

    a1 = atan2(z1,x1)
    a2 = acos((d2**2 - d3**2 + h**2)/(2 * d2 * h))

    # Shoulder angle (degrees)
    q2 = (a1 + a2) * 180/pi

    # Elbow angle (degrees)
    a3 = acos((d2**2 + d3**2 - h**2)/(2 * d2 * d3))
    q3 = 180 - a3 * 180/pi

    # Wrist angle (degrees) (add 5 deg because of the wrist-gripper offset)
    q4 = q0 - (q3 - q2) + gripOffset

    #  Add 15 deg because of the shoulder-elbow axis offset
    q2 = q2 + 15

    # Return values Base, Shoulder, Elbow, Wrist, Gripper
    angles_BSEWG = [ q1,   90-q2,   q3-90,  q4,     g0]

    # Check constraints
    for i in range(0,5):
        angles_BSEWG[i] = checkConstraints(angles_BSEWG[i], CST_ANGLE_MIN, CST_ANGLE_MAX)

    return(np.dot(10,angles_BSEWG).astype(int))

def LSSA_moveMotors(angles_BSEWG):

    # If the servos detect a current higher or equal than the value (mA) before reaching the requested positions they will halt and hold
    wrist.moveCH(angles_BSEWG[3], 1000)
    shoulder.moveCH(angles_BSEWG[1], 1600)
    elbow.moveCH(angles_BSEWG[2], 1600)
    base.moveCH(angles_BSEWG[0], 1000)
    gripper.moveCH(angles_BSEWG[4], 500)

    arrived = False
    issue = False

    # Check if they reached the requested position
    while arrived == False and issue == False:
        bStat = base.getStatus()
        sStat = shoulder.getStatus()
        eStat = elbow.getStatus()
        wStat = wrist.getStatus()
        gStat = gripper.getStatus()
        
        # If a status is None print message
        if (bStat is None or sStat is None or eStat is None or wStat is None or gStat is None):
            print("- Unknown status")
        # If the statuses aren't None check their values
        else:
            # If a servo is Outside limits, Stuck, Blocked or in Safe Mode before it reaches the requested position reset the servos and return issue
            if (int(bStat)>6 or int(sStat)>6 or int(eStat)>6 or int(wStat)>6 or int(gStat)>6):
                allMotors.setColorLED(lssc.LSS_LED_Red)
                wrist.moveCH(1100, 1000)
                shoulder.moveCH(-1150, 1600)
                elbow.moveCH(450, 1600)
                base.moveCH(0, 1000)
                time.sleep(2)
                allMotors.limp()
                allMotors.reset()
                time.sleep(2)
                allMotors.confirm()
                allMotors.setMaxSpeed(60)
                wrist.setMaxSpeed(120)
                print("- Issue detected")
                issue = 1
            # If all the servos are holding positions check if they have arrived
            elif (int(bStat)==6 and int(sStat)==6 and int(eStat)==6 and int(wStat)==6 and int(gStat)==6):
                bPos = base.getPosition()
                sPos = shoulder.getPosition()
                ePos = elbow.getPosition()
                wPos = wrist.getPosition()
                # If any position is None
                if (bPos is None or sPos is None or ePos is None or wPos is None):
                    print("- Unknown position")
                # If they are holding in a different position than requested one return issue
                elif (abs(int(bPos)-angles_BSEWG[0])>20 or abs(int(sPos)-angles_BSEWG[1])>20 or abs(int(ePos)-angles_BSEWG[2])>20 or abs(int(wPos)-angles_BSEWG[3])>20):
                    print("- Obstacle detected")
                    issue = 2
                else:
                    print("- Arrived\n")
                    arrived = True
                
    return(arrived, issue)

def CBtoXY(targetCBsq, params, color):

    wletterWeight = [-4, -3, -2, -1, 1, 2, 3, 4]
    bletterWeight = [4, 3, 2, 1, -1, -2, -3, -4]
    bnumberWeight = [8, 7, 6, 5, 4, 3, 2, 1]

    if targetCBsq[0] == 'k':
        x = 6 * params["sqSize"]
        y = 6 * params["sqSize"]
    else:
        if color:         # White -> Robot color = Black
            sqletter = bletterWeight[ord(targetCBsq[0]) - 97]
            sqNumber = bnumberWeight[int(targetCBsq[1]) - 1]
        else:                   # Black
            sqletter = wletterWeight[ord(targetCBsq[0]) - 97]
            sqNumber = int(targetCBsq[1])

        x = params["baseradius"] + params["cbFrame"] + params["sqSize"] * sqNumber - params["sqSize"]*1/2
        y = params["sqSize"] * sqletter - copysign(params["sqSize"]*1/2,sqletter)

    return(x,y)

def getGripperOffset(idx, color):
    offsets = [13, 11.5, 11.5, 11, 10.5, 10, 9.5, 9]
    
    if color:
        return offsets[8 - int(idx)]
    else:
        return offsets[int(idx) - 1]
    
def getZOffset(idx, color):
    offsets = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]
    if color:
        return offsets[8 - int(idx)]
    else:
        return offsets[int(idx) - 1]
    
def firstRotateWrist(square):
    specialSquare = {'a1','h1', 'b1', 'g1'}
    if square in specialSquare:
        return True
    else:
        return False

def testingExecuteMove(move, params, color):
    allMotors.setColorLED(lssc.LSS_LED_Cyan)
    z = params["cbHeight"] + params["pieceHeight"];
    angles_rest = (0,-1150,450,1100,0)
    gClose = -2.5
    gOpen = -9
    goDown = 0.6*params["pieceHeight"]
    gripState = gOpen
    for i in range(0,len(move),2):

        # Move to position (up)
        x, y = CBtoXY((move[i],move[i+1]), params, color)
        angles_BSEWG1 = LSS_IK([x, y, z + 1 + getZOffset(move[i + 1], color), gripState], 0)
        print("1) MOVE UP")
        arrived1,issue1 = LSSA_moveMotors(angles_BSEWG1)
        if not arrived1:
            print("move stage 1 failed issue:", issue1)
        time.sleep(1)
        
        # Go down
        angles_BSEWG2 = LSS_IK([x, y, z - 1 - goDown + getZOffset(move[i + 1], color), gripState], getGripperOffset(move[i + 1],color))
      
        if firstRotateWrist(move[i] + move[i + 1]):
            angleTmp = [ angles_BSEWG1[0], angles_BSEWG1[1], angles_BSEWG1[2], angles_BSEWG2[3], angles_BSEWG1[4]]
            LSSA_moveMotors(angleTmp)
        
        shoulder.setMaxSpeed(10)
        print("2) GO DOWN")
        arrived2,issue2 = LSSA_moveMotors(angles_BSEWG2)
        if not arrived2:
            print("move stage 2 failed isuue:", issue2)
        time.sleep(1)
        shoulder.setMaxSpeed(30)
        if (i/2)%2: # Uneven move (go lower to grab the piece)
            gripState = gOpen
            goDown = 0.6*params["pieceHeight"]
        else:       # Even move (go a little higher to drop the piece)
            gripState = gClose
            goDown = 0.1*params["pieceHeight"]
        time.sleep(1)

        # Close / Open the gripper
        gripper.moveCH(int(gripState*10), 500)
        time.sleep(1)
        print("3) CLOSE/OPEN the gripper\n")
        
        # Go up
        angles_BSEWG3 = LSS_IK([x, y, z + 1 + getZOffset(move[i + 1], color), gripState], getGripperOffset(move[i + 1], color))
        print("4) GO UP")
        arrived3,issue3 = LSSA_moveMotors(angles_BSEWG3)
        if not arrived3:
            print("move stage 3 failed issue:", issue3)
        time.sleep(1)
    # Go back to resting position and go limp
    print("5) REST")
    moveState,_ = LSSA_moveMotors(angles_rest)
    allMotors.limp()
    allMotors.setColorLED(lssc.LSS_LED_Black)
    return(moveState)
    
    
def executeMove(move, params, color, homography, cap, selectedCam):
    allMotors.setColorLED(lssc.LSS_LED_Cyan)
    z = params["cbHeight"] + params["pieceHeight"]
    angles_rest = (0,-1150,450,1100,0)
    gClose = -2.5
    gOpen = -9
    goDown = 0.6*params["pieceHeight"]
    gripState = gOpen
    
    for i in range(0,len(move),2):

        # Move to position (up)
        x, y = CBtoXY((move[i],move[i+1]), params, color)
        angles_BSEWG1 = LSS_IK([x, y, z + 1 + getZOffset(move[i + 1], color), gripState], 0)
        print("1) MOVE UP")
        arrived1,issue1 = LSSA_moveMotors(angles_BSEWG1)
        askPermision(angles_BSEWG1, arrived1, issue1, homography, cap, selectedCam)
        time.sleep(1)
        
        # Go down
        angles_BSEWG2 = LSS_IK([x, y, z - 1 - goDown + getZOffset(move[i + 1], color), gripState], getGripperOffset(move[i + 1],color))
        if firstRotateWrist(move[i] + move[i + 1]):
            angleTmp = [ angles_BSEWG1[0], angles_BSEWG1[1], angles_BSEWG1[2], angles_BSEWG2[3], angles_BSEWG1[4]]
            LSSA_moveMotors(angleTmp)
        shoulder.setMaxSpeed(10)
        print("2) GO DOWN")
        arrived2,issue2 = LSSA_moveMotors(angles_BSEWG2)
        askPermision(angles_BSEWG2, arrived2, issue2, homography, cap, selectedCam)
        time.sleep(1)
        shoulder.setMaxSpeed(30)
        if (i/2)%2: # Uneven move (go lower to grab the piece)
            gripState = gOpen
            goDown = 0.6*params["pieceHeight"]
        else:       # Even move (go a little higher to drop the piece)
            gripState = gClose
            goDown = 0.1*params["pieceHeight"]
        time.sleep(1)
        
        # Close / Open the gripper
        gripper.moveCH(int(gripState*10), 500)
        time.sleep(1)
        print("3) CLOSE/OPEN the gripper\n")
        
        # Go up
        angles_BSEWG3 = LSS_IK([x, y, z + 1 + getZOffset(move[i + 1], color), gripState], getGripperOffset(move[i + 1], color))
        print("4) GO UP")
        arrived3,issue3 = LSSA_moveMotors(angles_BSEWG3)
        askPermision(angles_BSEWG3, arrived3, issue3, homography, cap, selectedCam)
        time.sleep(1)
    # Go back to resting position and go limp
    print("5) REST")
    moveState,_ = LSSA_moveMotors(angles_rest)
    allMotors.limp()
    allMotors.setColorLED(lssc.LSS_LED_Black)

    return(moveState)

def askPermision(angles_BSEWG, arrived, issue, homography, cap, selectedCam):

    angles_rest = (0,-1150,450,1100,0)
    sec = 0

    while arrived == False:                                 # If the servos couldn't reach the requested position
        if issue == 1:
            inter.speak("reset")                            # If a servo exceeded a limit
        else:
            if sec == 0:
                inter.speak("excuse")                       # Play audio asking for permission
            else:
                inter.speak("please")
            allMotors.setColorLED(lssc.LSS_LED_Magenta)     # Change LEDs to Magenta
            LSSA_moveMotors(angles_rest)                    # Go to resting position
            allMotors.limp()
            sec = 0
            
        while arrived == False and sec < 3:                 # If the servos couldn't reach the requested position and we haven't waited 3 sec
            if vm.safetoMove(homography, cap, selectedCam) != 0 or sec == 2:  # Check if it is safe to move (vision code) or if we have waited 3 sec
                allMotors.setColorLED(lssc.LSS_LED_Cyan)    # If it is true we change LEDs back to cyan
                arrived,_ = LSSA_moveMotors(angles_BSEWG)   # And try moving the motors again
            else:
                time.sleep(1)                               # Wait one second
            sec = sec + 1

def winLED(allMotors):
    for i in range (0, 4):
        for j in range (1, 8):                  # Loop through each of the 8 LED color (black = 0, red = 1, ..., white = 7)
            allMotors.setColorLED(j)            # Set the color (session) of the LSS
            time.sleep(0.3)                     # Wait 0.3 seconds per color change
    allMotors.setColorLED(lssc.LSS_LED_Black)