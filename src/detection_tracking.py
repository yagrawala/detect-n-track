import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

########################
##  helper functions  ##
########################
def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")

def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

def visualise(frame, c, r, w=20, h=20, wait_time=30, operation='none', window_name='visual frame', save_frame=False, particles=None):
    if( operation=='none' ):
        frame_ww = frame
    elif( operation=='plot_particle' and particles.any()):
        frame_ww =  frame.copy()
        for particle in particles:
            # print particle[0], particle[1]
            point_radius = 1
            frame_ww = cv2.circle(frame_ww, (particle[0], particle[1]), point_radius, (0, 255, 0), 1)
    elif( operation=='plot_cr' ):
        point_radius = 5
        frame_ww = cv2.circle(frame, (c, r), point_radius, (0, 255, 0), 1)
    elif (w==20 and h==20):
        frame_ww = cv2.circle(frame, (c, r), w, (0, 255, 0), 1)
    else:
        frame_ww = cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)

    cv2.imshow(window_name, frame_ww)
    cv2.waitKey(wait_time) & 0xff

def save_frames(detector_type, frameCounter, frame, c, r, w=20, h=20):
    if detector_type=='of':
        frame_ww = frame
    elif (w==20 and h==20):
        frame_ww = cv2.circle(frame, (c, r), w, (0, 255, 0), 1)
    else:
        frame_ww = cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)
    output_name = sys.argv[3] + detector_type+ "/"+ str(frameCounter) + ".png"
    # print output_name
    cv2.imwrite(output_name, frame_ww)

#########################
##  tracker functions  ##
#########################

def camshift_tracker(v, file_name):
    # Interval of frames to visualise
    debug = False
    save_frames = False

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0

    # Read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # 1. Detect face in first frame
    # 2. Write track point for first frame
    # 3. Set the initial tracking window
    c,r,w,h = detect_one_face( frame )
    track_window = ( c,r,w,h )
    output.write( "{0},{1},{2}".format(frameCounter, c+w/2, r+h/2) )
    if(debug):
        print frameCounter, c, r, w, h
        visualise(frame, c, r, w, h)

    # calculate the HSV histogram in the window
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    frameCounter += 1
    while(1):
        # read next frame
        ret, frame = v.read()
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # write the output to the file
        x = int( ret[0][0] )
        y = int( ret[0][1] )
        output.write( "\n{0},{1},{2}".format(frameCounter, x, y  )  )

        if(debug):
            print "{0},{1},{2}".format(frameCounter, x, y  )
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            morphed_frame = cv2.polylines(frame, [pts], True, 255, 2)
            visualise(morphed_frame, x, y, operation='plot_cr', wait_time=60)
        if(save_frames):
            save_frames('camshift', frameCounter, frame, x, y, w, h)

        frameCounter += 1

    output.close()

def particle_tracker(v, file_name):
    # Interval of frames to visualise
    debug = False
    stepsize = 10

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    frameCounter = 0

    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # 1. Detect face in first frame
    # 2. Write track point for first frame
    # 3. Set the initial tracking window
    c,r,w,h = detect_one_face(frame)
    track_window = (c,r,w,h)
    output.write( "{0},{1},{2}".format(frameCounter, c+w/2, r+h/2) )
    if(debug):
        print frameCounter, c+w/2, r+h/2
        visualise(frame, c+w/2, r+h/2)

    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
    # c,r,w,h: obtain using detect_one_face()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    n_particles = 200

    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    # np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
    # visualise(frame, 0, 0, operation='plot_particle', particles=particles, wait_time=30)

    f = particleevaluator(hist_bp, particles.T) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)

    frameCounter = frameCounter + 1
    previous_to_previous = (c+w/2, r+h/2)
    previous = (c+w/2, r+h/2)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        # Particle motion model: uniform step (TODO: find a better motion model)
        x_diff = previous_to_previous[0] - previous[0]
        y_diff = previous_to_previous[1] - previous[1]
        vector = np.array([x_diff, y_diff], int)
        # print vector
        motion_vector = np.ones((n_particles, 2), int) * vector
        noise_vector = np.random.uniform(-stepsize, stepsize, particles.shape)
        final_vector = np.add( motion_vector, noise_vector, casting="unsafe")
        # motion_vector = np.ones((n_particles, 2), int) * init_pos
        np.add(particles, noise_vector, out=particles, casting="unsafe")
        # visualise(frame, 0, 0, operation='plot_particle', particles=particles, window_name='points_window')

        # Clip out-of-bounds particles( particles which have gone out of frame )
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int)

        f = particleevaluator(hist_bp, particles.T) # Evaluate particles

        # print f
        # print f.clip(1)
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        # print weights
        # return
        weights /= np.sum(weights)               # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
        output.write( "\n{0},{1},{2}".format(frameCounter, pos[0], pos[1]) )

        # if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
        if True: # If particle cloud degenerate:
            # print "resample"
            particles = particles[resample(weights),:]  # Resample particles according to weights
        # resample() function is provided for you

        previous_to_previous = previous
        previous = (pos[0], pos[1])
        if(debug):
            # print frameCounter, pos[0], pos[1]
            visualise(frame, pos[0], pos[1], operation='plot_cr')
        if( save_frames ):
            pass
            # save_frames('particle_detector', frameCounter, frame, x, y, w, h)

        frameCounter = frameCounter + 1

    output.close()

def kalman_tracker(v, file_name):
    # Interval of frames to visualise
    debug = False
    save_frame = False
    # visualise_interval = 10

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    frameCounter = 0

    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # 1. Detect face in first frame
    # 2. Write track point for first frame
    # 3. Set the initial tracking window
    c,r,w,h = detect_one_face(frame)
    track_window = (c,r,w,h)
    output.write( "{0},{1},{2}".format(frameCounter, c+w/2, r+h/2) )
    if(debug):
        print frameCounter, c+w/2, r+h/2
        visualise(frame, c+w/2, r+h/2)

    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman = cv2.KalmanFilter(4,2,0) # 4 state/hidden, 2 measurement, 0 control
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                        [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)      # respond faster to change and be less smooth
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state


    frameCounter += 1
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        prediction = kalman.predict()

        c,r,w,h = detect_one_face(frame)
        if (c!=0 and r!=0):
            state = np.array([c+w/2,r+h/2], dtype='float64')
            # print state
            posterior = kalman.correct(state)
            # print posterior
            # print "\n"
            if True:
                x = int( posterior[0] )
                y = int( posterior[1] )
            else:
                x = int( state[0] )
                y = int( state[1] )
        else:
            pass
        # select the prediction value here
        c = int(prediction[0][0])
        r = int(prediction[1][0])
        w = int(prediction[2][0])
        h = int(prediction[3][0])
        x = int(c+w/2)
        y = int(r+h/2)
        # print x,y
        # x = y = 0

        output.write( "\n{0},{1},{2}".format(frameCounter, x, y) )

        if(save_frame):
            save_frames('kalman', frameCounter, frame, x, y)
        if(debug):
            visualise(frame, x, y, operation='plot_cr', wait_time=60)

        frameCounter += 1

    output.close()

def good_feature_to_track(old_frame):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    # 1. Detect face in first frame
    # 2. Write track point for first frame
    # 3. Set the initial tracking window
    c,r,w,h = detect_one_face(old_frame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    feature_detection_mask = np.zeros(old_gray.shape, dtype='ubyte')
    feature_detection_mask[r+10:r+h-10, c+10:c+w-10] = 1
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = feature_detection_mask, **feature_params)
    return p0, old_gray, c, r, w, h

def of_tracker(v, file_name):
    # Interval of frames to visualise
    debug = False
    save_frame = False

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0

    # Read first frame
    ret ,old_frame = v.read()
    if ret == False:
        return

    p0, old_gray, c, r, w, h = good_feature_to_track(old_frame)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (w,h),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # track_window = (c,r,w,h)
    output.write( "{0},{1},{2}".format(frameCounter, c+w/2, r+h/2) )
    # if(debug):
        # print frameCounter, c, r, w, h
        # visualise(frame, c, r, w, h)

    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frameCounter += 1
    while(1):
        # read next frame
        ret, frame = v.read()
        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        count = sum_x = sum_y = 0
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            sum_x = sum_x + a
            sum_y = sum_y + b
            count += 1
            if(debug):
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame_copy = cv2.circle(frame.copy(),(a,b),5,color[i].tolist(),-1)

        # write the output to the file
        x = int( sum_x / count )
        y = int( sum_y / count )
        output.write( "\n{0},{1},{2}".format(frameCounter, x, y) )

        if(save_frame):
            save_frames('of', frameCounter, frame, 0, 0)
        if(debug):
            frame_copy = cv2.add(frame_copy,mask)
            cv2.imshow('frame',frame_copy)
            cv2.waitKey(60) & 0xff
            visualise(frame, x, y, operation='plot_cr', wait_time=10)

        # Now update the previous frame and previous points
        # p0_copy, old_gray_copy, c, r, w, h = good_feature_to_track(frame)
        # if(c!=0 and r!=0 and frameCounter>10 ):
        if False:
            p0 = np.reshape(p0_copy, (-1,1,2)) #p0_copy.reshape(-1,1,2)
            old_gray = old_gray_copy.copy()
        else:
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

        frameCounter += 1

    output.close()


if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")

'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
